# core.py
"""Main class for trust score calculation and reasoning graphs"""

import uuid
from typing import Dict, Optional, Type

from trustifai.config import Config
from trustifai.structures import (
    MetricContext,
    ReasoningGraph,
    ReasoningNode,
    ReasoningEdge,
)
from trustifai.services import ExternalService
from trustifai.metrics import (
    BaseMetric,
    EvidenceCoverageMetric,
    SemanticDriftMetric,
    EpistemicConsistencyMetric,
    SourceDiversityMetric,
    ConfidenceMetric,
)
from trustifai.metrics.calculators import ThresholdEvaluator
from trustifai.visualizer import GraphVisualizer


class Trustifai:
    """Main class for AI trust scoring"""

    # Central Registry for metrics
    _metric_registry: Dict[str, Type[BaseMetric]] = {
        "evidence_coverage": EvidenceCoverageMetric,
        "semantic_drift": SemanticDriftMetric,
        "consistency": EpistemicConsistencyMetric,
        "source_diversity": SourceDiversityMetric,
    }

    def __init__(self, context: MetricContext = None, config_path: str = "config_file.yaml"):
        self.context = context
        self.config = Config.from_yaml(config_path)
        self.service = ExternalService(self.config)
        self.threshold_evaluator = ThresholdEvaluator(self.config)

        self.metrics = {}
        if self.context:
            self._validate_context()
            self._compute_embeddings()
            self._init_metrics()

    @classmethod
    def register_metric(cls, name: str, metric_class: Type[BaseMetric]):
        """
        Register a new custom metric class.

        Args:
            name: The key used in config.yaml to refer to this metric.
            metric_class: A class inheriting from BaseMetric.
        """
        cls._metric_registry[name] = metric_class

    def _init_metrics(self):
        """Initialize only the metrics that have a non-zero weight (enabled)"""
        weights = self.config.weights

        self.metrics = {}

        # Iterate through the registry instead of hardcoded dict
        for metric_name, metric_cls in self._metric_registry.items():
            weight = getattr(weights, metric_name, 0.0)

            if weight > 0:
                self.metrics[metric_name] = metric_cls(
                    self.context, self.service, self.config
                )

    def _validate_context(self):
        required_attrs = ["answer", "query", "documents"]
        for attr in required_attrs:
            if not hasattr(self.context, attr):
                raise ValueError(f"Context missing required attribute: {attr}")

    def _compute_embeddings(self):
        """Compute embeddings if not present"""
        if (
            self.context.query_embeddings is None
            or self.context.answer_embeddings is None
            or self.context.document_embeddings is None
            or (
                hasattr(self.context.query_embeddings, "size")
                and self.context.query_embeddings.size == 0
            )
            or (
                hasattr(self.context.answer_embeddings, "size")
                and self.context.answer_embeddings.size == 0
            )
            or (
                isinstance(self.context.document_embeddings, list)
                and any(
                    (emb is None or (hasattr(emb, "size") and emb.size == 0))
                    for emb in self.context.document_embeddings
                )
            )
        ):

            self.context.query_embeddings = self.service.embedding_call(
                self.context.query
            )
            self.context.answer_embeddings = self.service.embedding_call(
                self.context.answer
            )
            self.context.document_embeddings = [
                self.service.embedding_call(self.service.extract_document(doc))
                for doc in self.context.documents
            ]

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Dict:
        """
        Generates a response using the configured LLM and calculates confidence metrics.

        Args:
            prompt: User query.
            system_prompt: System instruction.
            **kwargs: Extra LLM params.

        Returns:
            Dict containing 'response' text and 'metadata' with confidence scores.
        """
        kwargs["logprobs"] = True

        result = self.service.llm_call(
            prompt=prompt, system_prompt=system_prompt, **kwargs
        )

        if not result or result.get("response") is None:
            return {"response": None, "metadata": {"error": "LLM call failed"}}

        response_text = result["response"]
        logprobs = result.get("logprobs", [])

        confidence_result = ConfidenceMetric.calculate(
            logprobs, self.threshold_evaluator
        )

        # Log online metrics to MLflow
        if self.config.tracing and self.config.tracing.params.get('enabled', False):
            try:
                import mlflow
                if mlflow.active_run():
                    mlflow.log_metrics({"online/confidence_score": confidence_result["score"]})
                    mlflow.log_param("online/confidence_label", confidence_result["label"])
            except ImportError:
                pass

        return {
            "response": response_text,
            "metadata": {
                "confidence_score": confidence_result["score"],
                "confidence_label": confidence_result["label"],
                "confidence_details": confidence_result["details"],
                "logprobs_available": bool(logprobs),
            },
        }

    def evidence_coverage(self) -> Dict:
        return self.metrics.get("evidence_coverage").calculate().to_dict()

    def semantic_drift(self) -> Dict:
        return self.metrics.get("semantic_drift").calculate().to_dict()

    def epistemic_consistency(self) -> Dict:
        return self.metrics.get("consistency").calculate().to_dict()

    def source_diversity(self) -> Dict:
        return self.metrics.get("source_diversity").calculate().to_dict()

    def get_trust_score(self) -> Dict:
        """
        Calculates the overall trust score using active metrics and normalized weights.
        """
        if not self.context.documents:
            return {
                "score": 0.0,
                "label": "Unreliable",
                "details": "No source documents available.",
            }

        # Calculate all ACTIVE metrics
        metrics_data = {k: m.calculate().to_dict() for k, m in self.metrics.items()}

        # Weighted Sum
        score = 0.0
        weights_dict = self.config.weights.model_dump()

        for metric_key, result in metrics_data.items():
            # Get normalized weight
            w = weights_dict.get(metric_key, 0.0)

            # Specific handling for source diversity score scaling
            metric_score = result["score"]
            if metric_key == "source_diversity":
                metric_score = min(1.0, metric_score / 3)

            score += w * metric_score

        thresholds = self.config.thresholds
        if score >= thresholds.RELIABLE_TRUST:
            decision = "RELIABLE"
        elif score >= thresholds.ACCEPTABLE_TRUST:
            decision = "ACCEPTABLE (WITH CAUTION)"
        else:
            decision = "UNRELIABLE"

        # Log categorized metrics to MLflow
        if self.config.tracing and self.config.tracing.params.get('enabled', False):
            try:
                offline_metric_keys = set(self._metric_registry.keys())
                self.service.log_metrics_by_category(metrics_data, score, decision, offline_metric_keys)
            except ImportError:
                pass

        return {
            "score": round(score, 2),
            "label": decision,
            "details": metrics_data,
        }

    def build_reasoning_graph(
        self, trust_score: Optional[Dict] = None
    ) -> ReasoningGraph:
        trace_id = str(uuid.uuid4())
        if trust_score is None:
            trust_score = self.get_trust_score()

        score = trust_score["score"]
        decision = trust_score["label"]
        weights = self.config.weights
        thresholds = self.config.thresholds

        nodes = self._build_nodes(trust_score, score, decision, weights, thresholds)
        edges = self._build_edges(trust_score["details"])

        return ReasoningGraph(trace_id, nodes, edges)

    def _build_nodes(
        self, trust_score: Dict, score: float, decision: str, weights, thresholds
    ) -> list:
        details = trust_score["details"]

        # Metric Nodes (only active ones)
        metric_nodes = [
            ReasoningNode(
                node_id=key,
                node_type="metric",
                name=key.replace("_", " ").title(),
                inputs={},
                outputs={},
                score=val["score"],
                label=val["label"],
                details=val["details"],
            )
            for key, val in details.items()
        ]

        # Aggregation Node
        agg_node = ReasoningNode(
            node_id="trust_aggregation",
            node_type="aggregation",
            name="Trust Score",
            inputs={"weights": weights.model_dump()},
            outputs={"final_score": score},
            score=score,
            label=decision,
            details={"explanation": "Weighted aggregation of active metrics"},
        )

        # Decision Node
        decision_node = ReasoningNode(
            node_id="final_decision",
            node_type="decision",
            name=f"Decision: {decision}",
            inputs={
                "thresholds": {
                    "reliable": thresholds.RELIABLE_TRUST,
                    "warning": thresholds.ACCEPTABLE_TRUST,
                }
            },
            outputs={"decision": decision},
            score=score,
            label=decision,
            details="Final trust classification based on aggregated score.",
        )

        return metric_nodes + [agg_node, decision_node]

    @staticmethod
    def _build_edges(active_metrics: Dict) -> list:
        """Dynamically build edges for active metrics only"""
        edges = []
        for metric_key in active_metrics.keys():
            edges.append(ReasoningEdge(metric_key, "trust_aggregation", ""))

        edges.append(ReasoningEdge("trust_aggregation", "final_decision", "decides"))
        return edges

    def visualize(self, graph: ReasoningGraph, graph_type: str = "pyvis"):
        visualizer = GraphVisualizer(graph, self.config)
        return visualizer.visualize(graph_type)
