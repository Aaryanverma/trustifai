# base.py
"""Base metric class definition"""
from abc import ABC, abstractmethod
from trustifai.structures import MetricResult, MetricContext
from trustifai.services import ExternalService
from trustifai.config import Config
from trustifai.metrics.calculators import CosineSimCalculator, DocumentExtractor, ThresholdEvaluator

class BaseMetric(ABC):
    """Base class for all metrics"""

    def __init__(self, service: ExternalService, config: Config):
        self.service = service
        self.config = config
        self.cosine_calc = CosineSimCalculator()
        self.doc_extractor = DocumentExtractor(service)
        self.threshold_evaluator = ThresholdEvaluator(config)

    @abstractmethod
    def calculate(self, context: MetricContext) -> MetricResult:
        """Synchronous calculation per request context"""
        pass

    async def a_calculate(self, context: MetricContext) -> MetricResult:
        """Asynchronous calculation. Override if async logic is needed natively."""
        return self.calculate(context)