# offline_metrics.py
"""Offline Metric calculators"""

import json
import re
import numpy as np
from typing import List
from nltk.tokenize import sent_tokenize

from trustifai.config import Config
from trustifai.structures import (
    MetricContext,  
    SpanSchema,
    MetricResult,
    SpanCheckResult,
    RerankerResult,
    TrustLevel,
)
from trustifai.services import ExternalService
from trustifai.metrics.calculators import SourceIdentifier
from trustifai.metrics.base import BaseMetric
import asyncio
import logging
import threading
import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger(__name__)

class EvidenceCoverageMetric(BaseMetric):
    def __init__(self, service: ExternalService, config: Config):
        super().__init__(service, config)
        self.strategy = self._select_strategy()

    def _select_strategy(self):
        metric_cfg = next((m for m in self.config.metrics if m.type == "evidence_coverage"), None)
        explicit_strategy = metric_cfg.params.get("strategy") if metric_cfg else None

        if explicit_strategy == "reranker":
            if self.config.reranker and self.config.reranker.type:
                return RerankerBasedEvidenceStrategy(self.service, self.config)
            logger.warning("Reranker configured but missing globally. Falling back to LLM.")
            return LLMBasedEvidenceStrategy(self.service, self.config)
        elif explicit_strategy == "llm":
            return LLMBasedEvidenceStrategy(self.service, self.config)

        if self.config.reranker is not None and self.config.reranker.type:
            return RerankerBasedEvidenceStrategy(self.service, self.config)

        return LLMBasedEvidenceStrategy(self.service, self.config)

    def calculate(self, context: MetricContext) -> MetricResult:
        spans = sent_tokenize(context.answer)
        if not spans:
            return MetricResult(score=0.0, label="Empty Answer", details={"sentences_checked": 0})
        return self.strategy.calculate(context, spans)
        
    async def a_calculate(self, context: MetricContext) -> MetricResult:
        return self.calculate(context) # Currently blocking in batch, can be optimized


class SemanticDriftMetric(BaseMetric):
    def calculate(self, context: MetricContext) -> MetricResult:
        if context.documents is None or len(context.documents) == 0:
            return MetricResult(
                score=0.0, label="No Documents", details={"docs_checked": 0}
            )

        all_sentences = []
        for doc in context.documents:
            text = self.service.extract_document(doc)
            all_sentences.extend(sent_tokenize(text))

        if not all_sentences:
            return MetricResult(score=0.0, label="Empty Context", details={})

        answer_emb = np.atleast_2d(context.answer_embeddings)
        query_emb = np.atleast_2d(context.query_embeddings)

        # Embed all document sentences at once (batched)
        try:
            embedding_result = self.service.embedding_call_batch(all_sentences)
        except Exception as e:
            logger.exception(f"Error during embedding call batch: {e}")
            return MetricResult(score=0.0, label="Embedding Error", details={"error": str(e), "sentences_checked": len(all_sentences)}, execution_metadata={"total_cost_usd": 0.0})
        
        sentence_embeddings = embedding_result['embedding'] if embedding_result and 'embedding' in embedding_result else []
        cost = embedding_result["cost"] if embedding_result else 0.0

        best_score = 0.0
        best_sentence = ""

        for sentence, sent_emb in zip(all_sentences, sentence_embeddings):
            sent_emb = np.atleast_2d(sent_emb)

            sim = max(
                self.cosine_calc.calculate(answer_emb, sent_emb),
                self.cosine_calc.calculate(query_emb, sent_emb),
            )

            if sim > best_score:
                best_score = sim
                best_sentence = sentence

        label, explanation = self.threshold_evaluator.evaluate_drift(best_score)

        return MetricResult(
            score=best_score,
            label=label,
            details={
                "explanation": explanation,
                "total_documents": len(context.documents),
                "total_sentences_checked": len(all_sentences),
                "best_matching_sentence": best_sentence[:150] + " ... [truncated]" if len(best_sentence) > 150 else best_sentence,
                "execution_metadata": {"total_cost_usd": cost}
            },
        )


class EpistemicConsistencyMetric(BaseMetric):
    def calculate(self, context: MetricContext) -> MetricResult:
        """Runs the async generation in a completely safe isolated thread to prevent event-loop clashes"""
        if self.config.k_samples == 0:
            return self._create_stable_result()

        result_container = []
        
        def run_async_isolated():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                res, cost = loop.run_until_complete(self._generate_samples_async(context))
                result_container.append((res, cost))
            finally:
                loop.close()

        # Isolate completely from any running event loops (FastAPI, Celery, Notebooks)
        thread = threading.Thread(target=run_async_isolated)
        thread.start()
        thread.join()

        samples, cost = result_container[0] if result_container else ([], 0.0)
        similarities = self._calculate_similarities(samples, context)
        if not similarities:
            return self._create_unreliable_result(cost)

        score = float(np.mean(similarities))
        std = float(np.std(similarities)) if len(similarities) > 1 else 0.0
        ci_95 = 1.96 * (std / np.sqrt(self.config.k_samples)) 

        return self._format_result(score, std, ci_95, cost)

    async def a_calculate(self, context: MetricContext) -> MetricResult:
        """Native async pipeline for non-blocking server applications"""
        if self.config.k_samples == 0:
            return self._create_stable_result()

        samples, cost = await self._generate_samples_async(context)
        similarities = self._calculate_similarities(samples, context)

        if not similarities:
            return self._create_unreliable_result(cost)

        score = float(np.mean(similarities))
        std = float(np.std(similarities)) if len(similarities) > 1 else 0.0
        ci_95 = 1.96 * (std / np.sqrt(self.config.k_samples)) 

        return self._format_result(score, std, ci_95, cost)

    async def _generate_samples_async(self, context: MetricContext):
        temperature_options = [0.7, 0.8, 0.9, 1.0]
        temps = np.random.choice(temperature_options, self.config.k_samples)

        tasks = [self.service.llm_call_async(prompt=context.query, temperature=float(temp)) for temp in temps]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_responses = [r["response"] for r in responses if isinstance(r, dict) and r.get("response")]
        cost = sum([r.get("cost", 0.0) for r in responses if isinstance(r, dict)])
        
        return valid_responses, cost

    def _calculate_similarities(self, samples: List[str], context: MetricContext) -> List[float]:
        main_emb = np.atleast_2d(np.array(context.answer_embeddings))
        try:
            sample_embeddings = self.service.embedding_call_batch(samples)['embedding']
        except Exception as e:
            logger.exception(f"Error during embedding call batch: {e}")
            return []

        similarities = []
        for sample_emb_list in sample_embeddings:
            sample_emb = np.atleast_2d(np.array(sample_emb_list))
            if sample_emb is not None and sample_emb.size > 0:
                sim = self.cosine_calc.calculate(main_emb, sample_emb)
                similarities.append(sim)
        return similarities

    def _format_result(self, score: float, std: float, ci_95: float, cost: float) -> MetricResult:
        label, explanation = self.threshold_evaluator.evaluate_consistency(score)
        return MetricResult(
            score=score, label=label,
            details={"explanation": explanation, "std_dev": round(std, 2), "uncertainty": round(ci_95, 2)},
            execution_metadata={"total_cost_usd": cost}
        )

    def _create_stable_result(self) -> MetricResult:
        return MetricResult(score=1.0, label=TrustLevel.STABLE.value, details={"explanation": "Assumed stable."}, execution_metadata={"total_cost_usd": 0.0})
    def _create_unreliable_result(self, cost: float) -> MetricResult:
        return MetricResult(score=0.0, label=TrustLevel.UNRELIABLE.value, details={"explanation": "No valid samples."}, execution_metadata={"total_cost_usd": cost})


class SourceDiversityMetric(BaseMetric):
    def calculate(self, context: MetricContext) -> MetricResult:
        if context.documents is None or len(context.documents) == 0:
            return MetricResult(
                score=0.0, label="No Trust", details={"unique_sources": 0}
            )

        source_identifier = SourceIdentifier()
        source_ids = {
            source_identifier.resolve_source_id(doc, self.service)
            for doc in context.documents
        }

        count = len(source_ids)
        total_docs = len(context.documents)
        
        # Check if low diversity is justified (only 1 relevant doc)
        relevant_docs_count = self._count_relevant_documents(context)
        is_justified_single_source = (count == 1 and relevant_docs_count <= 1)
        
        normalized_score = self._calculate_normalized_score(
            count, total_docs, is_justified_single_source
        )
        label, explanation = self.threshold_evaluator.evaluate_diversity(normalized_score)
        
        # Override explanation if single source is justified
        if is_justified_single_source:
            explanation = "Single source justified: only one document contains relevant information"
            label = "Acceptable"

        return MetricResult(
            score=normalized_score,
            label=label,
            details={
                "explanation": explanation,
                "unique_sources": count,
                "total_documents": total_docs,
                "relevant_documents": relevant_docs_count,
                "justified_single_source": is_justified_single_source,
            },
        )

    def _count_relevant_documents(self, context: MetricContext) -> int:
        """Count documents semantically relevant to the query."""
        if context.query is None or context.documents is None or len(context.documents) == 0:
            return 0

        try:
            query_emb = np.atleast_2d(context.query_embeddings)
        except Exception as e:
            logger.exception(f"Error creating query embedding: {e}")
            return 0

        relevance_threshold = 0.5  # Configurable via config if needed
        
        relevant_count = 0
        for doc_emb in context.document_embeddings:
            try:
                doc_emb = np.atleast_2d(doc_emb)
                similarity = self.cosine_calc.calculate(query_emb, doc_emb)
                if similarity >= relevance_threshold:
                    relevant_count += 1
            except Exception as e:
                logger.exception(f"Error calculating similarity for document embedding: {e}")
                continue

        return max(relevant_count, 1)  # At least 1 to avoid division by zero

    @staticmethod
    def _calculate_normalized_score(
        count: int, total: int, is_justified: bool = False
    ) -> float:
        if total == 0:
            return 0.0
        
        # If single source is justified, don't penalize
        if is_justified:
            return 0.8  # High score, but not perfect (room for improvement)
        
        diversity_ratio = count / total
        count_score = 1 - np.exp(-count / 2)
        return 0.6 * diversity_ratio + 0.4 * count_score


class LLMBasedEvidenceStrategy(BaseMetric):
    def calculate(self, context: MetricContext, spans: List[str]) -> MetricResult:
        if context.documents is None or len(context.documents) == 0:
            return MetricResult(
                score=0.0, label="No Documents", details={"sentences_checked": 0}
            )

        extracted_docs = [
            self.service.extract_document(doc) for doc in context.documents
        ]
        result = self._verify_spans_with_llm(spans, extracted_docs)

        score = (
            result.supported_count / result.total_count
            if result.total_count > 0
            else 0.0
        )
        label, explanation = self.threshold_evaluator.evaluate_grounding(score)

        return MetricResult(
            score=score,
            label=label,
            details={
                "explanation": explanation,
                "strategy": "LLM",
                "total_sentences": result.total_count,
                "supported_sentences": result.supported_count,
                "unsupported_sentences": result.unsupported_spans,
                "failed_checks": result.failed_count,
            },
            execution_metadata={"total_cost_usd": result.cost}
        )

    def _verify_spans_with_llm(
        self, spans: List[str], extracted_docs: List[str]
    ) -> SpanCheckResult:
        supported = 0
        failed_checks = 0
        fail_reason = None
        unsupported_spans = []

        prompts = [self._build_verification_prompt(span, extracted_docs) for span in spans]

        try:
            batch_results = self.service.llm_call_batch(prompts=prompts, response_format=SpanSchema)
        except Exception as e:
            logger.exception(f"Error calling LLM batch: {e}")
            return SpanCheckResult(0, spans, len(spans), f"Batch LLM call failed: {e}", len(spans), 0.0)

        if not batch_results or not batch_results.get("response"):
            return SpanCheckResult(0, spans, len(spans), "Batch LLM call failed", len(spans), batch_results.get("cost", 0.0))

        responses = batch_results["response"]

        for i, response_content in enumerate(responses):
            if not response_content:
                failed_checks += 1
                continue
                
            try:
                match = re.search(r'\{.*\}', response_content, re.DOTALL)
                clean_content = match.group(0) if match else response_content
                result = json.loads(clean_content)
                spans_result = result.get("spans", [])
                if spans_result and spans_result[0].get("supported", False):
                    supported += 1
                else:
                    unsupported_spans.append(spans[i])

            except Exception as e:
                failed_checks += 1
                fail_reason = f"Parse error: {e}"

        return SpanCheckResult(
            supported, unsupported_spans, failed_checks, fail_reason, len(spans), batch_results.get("cost", 0.0)
        )

    @staticmethod
    def _build_verification_prompt(span: str, docs: List[str]) -> str:
        return f"""Evaluate if the answer span is factually supported by the provided documents.
        Think thoroughly and reason about the evidence in the documents before answering.
        DOCUMENTS: {docs}
        ANSWER SPAN TO CHECK: {span}
        Return ONLY a JSON object: {{"spans": [{{"index": 0, "supported": true/false, "answer": "<answer_span>"}}]}}"""


class RerankerBasedEvidenceStrategy(BaseMetric):
    TRUST_THRESHOLD = 0.85
    LOW_RISK_THRESHOLD = 0.49
    GLOBAL_PASS_THRESHOLD = 0.5

    def calculate(self, context: MetricContext, spans: List[str]) -> MetricResult:
        if context.documents is None or len(context.documents) == 0:
            return MetricResult(
                score=0.0, label="No Documents", details={"sentences_checked": 0}
            )

        extracted_docs = [
            self.service.extract_document(doc) for doc in context.documents
        ]
        combined_docs = " ".join(extracted_docs)
        reranker_result = self._check_with_reranker(combined_docs, spans)
        label, explanation = self.threshold_evaluator.evaluate_grounding(
            reranker_result.mean_score
        )

        return MetricResult(
            score=reranker_result.mean_score,
            label=label,
            details={
                "explanation": explanation,
                "strategy": "Reranker",
                "total_sentences": len(spans),
                "fully_supported_sentences": reranker_result.fully_supported,
                "partially_supported_sentences": reranker_result.partially_supported,
                "detailed_scores": reranker_result.detailed_results,
            },
        )

    def _check_with_reranker(
        self, document_text: str, spans: List[str]
    ) -> RerankerResult:
        response_data = self.service.reranker_call(query=document_text, documents=spans)
        results = [None] * len(spans)
        fully_supported = 0
        partial_supported = []

        for item in response_data:
            idx = item["index"]
            score = item["relevance_score"]
            label = self._classify_trust_level(score)
            results[idx] = {
                "sentence": spans[idx],
                "trust_score": round(score, 2),
                "label": label,
            }

            if label == "Trusted":
                fully_supported += 1
            elif label == "Low Risk":
                partial_supported.append(spans[idx])

        score_list = [r["trust_score"] for r in results]
        mean_score = np.mean(score_list) if score_list else 0.0
        global_pass = (
            "Pass"
            if min(score_list, default=0) > self.GLOBAL_PASS_THRESHOLD
            else "Fail"
        )

        return RerankerResult(
            mean_score, global_pass, fully_supported, partial_supported, results
        )

    def _classify_trust_level(self, score: float) -> str:
        if score >= self.TRUST_THRESHOLD:
            return "Trusted"
        elif score > self.LOW_RISK_THRESHOLD:
            return "Low Risk"
        else:
            return "High Risk"