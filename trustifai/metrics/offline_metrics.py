# offline_metrics.py
"""Offline Metric calculators"""

import json
import numpy as np
from typing import List
from nltk.tokenize import sent_tokenize
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from trustifai.config import Config
from trustifai.structures import (
    MetricContext,  
    SpanSchema,
    MetricResult,
    SpanCheckResult,
    RerankerResult,
    TrustLevel,
)
from trustifai.services import ExternalService, is_notebook
from trustifai.metrics.calculators import SourceIdentifier
from trustifai.metrics.base import BaseMetric
import asyncio
import logging

logger = logging.getLogger(__name__)

class EvidenceCoverageMetric(BaseMetric):
    def __init__(
        self, context: MetricContext, service: ExternalService, config: Config
    ):
        super().__init__(context, service, config)
        self.strategy = self._select_strategy()

    def _select_strategy(self):
        """
        Selects strategy based on precedence:
        1. Explicit 'strategy' param in metrics config ('reranker' | 'llm')
        2. Implicit availability of reranker config
        """
        # 1. Locate specific metric configuration
        metric_cfg = next(
            (m for m in self.config.metrics if m.type == "evidence_coverage"), None
        )
        explicit_strategy = metric_cfg.params.get("strategy") if metric_cfg else None

        # 2. Handle Explicit Strategy
        if explicit_strategy == "reranker":
            if self.config.reranker and self.config.reranker.type:
                return RerankerBasedEvidenceStrategy(
                    self.context, self.service, self.config
                )
            logger.warning(
                "Warning: Evidence Coverage strategy set to 'reranker' but global reranker config is missing/disabled. Falling back to LLM."
            )
            return LLMBasedEvidenceStrategy(self.context, self.service, self.config)

        elif explicit_strategy == "llm":
            return LLMBasedEvidenceStrategy(self.context, self.service, self.config)

        # 3. Handle Implicit Strategy (Default)
        if self.config.reranker is not None and self.config.reranker.type:
            return RerankerBasedEvidenceStrategy(
                self.context, self.service, self.config
            )

        return LLMBasedEvidenceStrategy(self.context, self.service, self.config)

    def calculate(self) -> MetricResult:
        spans = sent_tokenize(self.context.answer)
        if not spans:
            return MetricResult(
                score=0.0, label="Empty Answer", details={"sentences_checked": 0}
            )
        return self.strategy.calculate(spans)


class SemanticDriftMetric(BaseMetric):
    def calculate(self) -> MetricResult:
        if not self.context.documents:
            return MetricResult(
                score=0.0, label="No Documents", details={"docs_checked": 0}
            )

        mean_doc_emb = np.mean(self.context.document_embeddings, axis=0)
        score = self.cosine_calc.calculate(self.context.answer_embeddings, mean_doc_emb)
        label, explanation = self.threshold_evaluator.evaluate_drift(score)

        return MetricResult(
            score=score,
            label=label,
            details={
                "explanation": explanation,
                "total_documents": len(self.context.documents),
            },
        )


class EpistemicConsistencyMetric(BaseMetric):
    def calculate(self) -> MetricResult:
        if self.config.k_samples == 0:
            return self._create_stable_result()

        samples = self._generate_samples()
        similarities = self._calculate_similarities(samples)

        if not similarities:
            return self._create_unreliable_result()

        score = float(np.mean(similarities))
        std = float(np.std(similarities)) if len(similarities) > 1 else 0.0
        ci_95 = 1.96 * (std / np.sqrt(self.config.k_samples)) #In a normal distribution, 95% of values fall within ±1.96 standard deviations of the mean

        return self._format_result(score, std, ci_95)

    def _generate_samples(self) -> List[str]:

        temperature_options = [0.7, 0.8, 0.9, 1.0]
        temps = np.random.choice(temperature_options, self.config.k_samples)

        async def gather_samples():
            tasks = [
                self.service.llm_call_async(
                    prompt=self.context.query, temperature=float(temp)
                )
                for temp in temps
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [
                r["response"]
                for r in responses
                if isinstance(r, dict) and r.get("response")
            ]

        if is_notebook():
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(gather_samples())
        else:
            return asyncio.run(gather_samples())

    def _calculate_similarities(self, samples: List[str]) -> List[float]:
        main_emb = np.atleast_2d(
            np.array(self.service.embedding_call(self.context.answer))
        )
        similarities = []
        for sample in samples:
            sample_emb = np.atleast_2d(np.array(self.service.embedding_call(sample)))
            if sample_emb is not None and sample_emb.size > 0:
                sim = self.cosine_calc.calculate(main_emb, sample_emb)
                similarities.append(sim)
        return similarities

    def _format_result(self, score: float, std: float, ci_95: float) -> MetricResult:
        label, explanation = self.threshold_evaluator.evaluate_consistency(score)
        return MetricResult(
            score=score,
            label=label,
            details={
                "explanation": explanation,
                "std_dev": round(std, 2),
                "uncertainty": round(ci_95, 2),
            }
        )

    def _create_stable_result(self) -> MetricResult:
        return MetricResult(
            score=1.0,
            label=TrustLevel.STABLE.value,
            details={
                "explanation": "No samples generated; assumed stable.",
                "std_dev": 0.0,
                "uncertainty": 0.0,
            },
        )

    def _create_unreliable_result(self) -> MetricResult:
        return MetricResult(
            score=0.0,
            label=TrustLevel.UNRELIABLE.value,
            details={"explanation": "No valid samples for comparison.", "std_dev": 0.0, "uncertainty": 0.0},
        )


class SourceDiversityMetric(BaseMetric):
    def calculate(self) -> MetricResult:
        if not self.context.documents:
            return MetricResult(
                score=0.0, label="No Trust", details={"unique_sources": 0}
            )

        source_identifier = SourceIdentifier()
        source_ids = {
            source_identifier.resolve_source_id(doc, self.service)
            for doc in self.context.documents
        }

        count = len(source_ids)
        total_docs = len(self.context.documents)
        
        # Check if low diversity is justified (only 1 relevant doc)
        relevant_docs_count = self._count_relevant_documents()
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

    def _count_relevant_documents(self) -> int:
        """Count documents semantically relevant to the query."""
        if not self.context.query or not self.context.documents:
            return len(self.context.documents)
        
        query_emb = np.atleast_2d(self.context.query_embeddings)
        relevance_threshold = 0.5  # Configurable via config if needed
        
        relevant_count = 0
        for doc_emb in self.context.document_embeddings:
            doc_emb = np.atleast_2d(doc_emb)
            similarity = self.cosine_calc.calculate(query_emb, doc_emb)
            if similarity >= relevance_threshold:
                relevant_count += 1
        
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
    def calculate(self, spans: List[str]) -> MetricResult:
        if not self.context.documents:
            return MetricResult(
                score=0.0, label="No Documents", details={"sentences_checked": 0}
            )

        extracted_docs = [
            self.service.extract_document(doc) for doc in self.context.documents
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
        )

    def _verify_spans_with_llm(
        self, spans: List[str], extracted_docs: List[str]
    ) -> SpanCheckResult:
        supported = 0
        failed_checks = 0
        fail_reason = None
        unsupported_spans = []

        for span in spans:
            prompt = self._build_verification_prompt(span, extracted_docs)
            response = self.service.llm_call(prompt=prompt, response_format=SpanSchema)

            if not response:
                failed_checks += 1
                fail_reason = "No response"
                continue
            try:
                result = json.loads(response["response"])
                spans_result = result.get("spans", [])
                if spans_result and spans_result[0].get("supported", False):
                    supported += 1
                else:
                    unsupported_spans.append(span)

            except Exception as e:
                failed_checks += 1
                fail_reason = f"Parse error: {e}"

        return SpanCheckResult(
            supported, unsupported_spans, failed_checks, fail_reason, len(spans)
        )

    @staticmethod
    def _build_verification_prompt(span: str, docs: List[str]) -> str:
        return f"""Evaluate if the answer span is factually supported by the provided documents.
        DOCUMENTS: {docs}
        ANSWER SPAN TO CHECK: {span}
        Return ONLY a JSON object: {{"spans": [{{"index": 0, "supported": true/false, "answer": "<answer_span>"}}]}}"""


class RerankerBasedEvidenceStrategy(BaseMetric):
    TRUST_THRESHOLD = 0.85
    LOW_RISK_THRESHOLD = 0.49
    GLOBAL_PASS_THRESHOLD = 0.5

    def calculate(self, spans: List[str]) -> MetricResult:
        if not self.context.documents:
            return MetricResult(
                score=0.0, label="No Documents", details={"sentences_checked": 0}
            )

        extracted_docs = [
            self.service.extract_document(doc) for doc in self.context.documents
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
