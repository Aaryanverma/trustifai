# base.py
"""Base metric class definition"""
from abc import ABC, abstractmethod
from trustifai.structures import MetricResult, MetricContext
from trustifai.services import ExternalService
from trustifai.config import Config
from trustifai.metrics.calculators import CosineSimCalculator, DocumentExtractor, ThresholdEvaluator

class BaseMetric(ABC):
    """Base class for all metrics"""

    def __init__(self, context: MetricContext, service: ExternalService, config: Config):
        self.context = context
        self.service = service
        self.config = config
        self.cosine_calc = CosineSimCalculator()
        self.doc_extractor = DocumentExtractor(service)
        self.threshold_evaluator = ThresholdEvaluator(config)

    @abstractmethod
    def calculate(self) -> MetricResult:
        pass