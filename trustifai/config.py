# config.py
"""
Configuration management.
"""

import yaml
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, model_validator

# --- Sub-configurations ---

class ModelConfig(BaseModel):
    """Configuration for LLM, Embeddings, or Reranker models"""
    type: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class TracingConfig(BaseModel):
    type: str = "default"
    params: Dict[str, Any] = Field(default_factory=dict)

class MetricConfigItem(BaseModel):
    """Represents a single metric entry in the YAML list"""
    type: str
    enabled: bool = True # Added enabled flag
    params: Dict[str, Any] = Field(default_factory=dict)

class ScoreWeightItem(BaseModel):
    """Represents a single weight entry in the YAML list"""
    type: str
    params: Dict[str, float]

# --- Derived Logic Classes ---

class MetricThresholds(BaseModel):
    """Thresholds for metrics"""
    STRONG_GROUNDING: float = 0.85
    PARTIAL_GROUNDING: float = 0.60
    STRONG_ALIGNMENT: float = 0.85
    PARTIAL_ALIGNMENT: float = 0.60
    STABLE_CONSISTENCY: float = 0.85
    FRAGILE_CONSISTENCY: float = 0.60
    HIGH_DIVERSITY: float = 0.85
    MODERATE_DIVERSITY: float = 0.60
    RELIABLE_TRUST: float = 0.80
    ACCEPTABLE_TRUST: float = 0.60
    HIGH_CONFIDENCE: float = 0.90
    MEDIUM_CONFIDENCE: float = 0.70
    
    model_config = {"extra": "allow"}

class TrustWeights(BaseModel):
    """
    Weights derived from the 'score_weights' list in YAML.
    Defaults to 0.0 to allow for disabled metrics.
    """
    evidence_coverage: float = 0.0
    semantic_drift: float = 0.0
    consistency: float = 0.0
    source_diversity: float = 0.0
    confidence: float = 0.0

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_sum(self):
        # Validate sum is approx 1.0 (unless all are 0, which implies misconfig but is technically valid logic)
        # We filter out non-numeric values just in case, though typing should handle it
        values = [v for v in self.model_dump().values() if isinstance(v, (int, float))]
        total = sum(values)
        if total > 0 and not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must normalize to 1.0, got {total}")
        return self

# --- Main Configuration ---

class Config(BaseModel):
    """Global Configuration Object"""
    env_file: str
    tracing: Optional[TracingConfig] = None
    llm: ModelConfig
    embeddings: ModelConfig
    reranker: Optional[ModelConfig] = None
    metrics: List[MetricConfigItem]
    score_weights: List[ScoreWeightItem]
    
    # Derived fields (populated post-init)
    thresholds: MetricThresholds = Field(default_factory=MetricThresholds)
    weights: TrustWeights = Field(default_factory=TrustWeights)

    # Global params
    k_samples: int = 3
    batch_size: int = 3

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, 'r') as f:
            raw_data = yaml.safe_load(f)
        
        config = cls(**raw_data)
        config._parse_thresholds()
        config._parse_weights()
        return config

    def _parse_thresholds(self):
        """Flatten metric params into Thresholds object"""
        t_data = {}
        for metric in self.metrics:
            t_data.update(metric.params)
        self.thresholds = MetricThresholds(**t_data)

    def _parse_weights(self):
        """
        Flatten weight params into Weights object and Normalize.
        1. Identify enabled metrics.
        2. Filter weights for enabled metrics.
        3. Normalize so sum is 1.0.
        """
        # Map of metric_type -> enabled_status
        enabled_map = {m.type: m.enabled for m in self.metrics}
        
        raw_weights = {}
        for item in self.score_weights:
            # Only consider if the metric is explicitly enabled in the metrics list
            is_enabled = enabled_map.get(item.type, True)
            
            if is_enabled:
                raw_weights[item.type] = item.params.get("weight", 0.0)
            else:
                raw_weights[item.type] = 0.0

        # Normalize
        total_raw = sum(raw_weights.values())
        if total_raw > 1.0:
            raise ValueError(f"Weights must normalize to 1.0, got {total_raw}")
        normalized_weights = {}
        
        if total_raw > 0:
            for k, v in raw_weights.items():
                normalized_weights[k] = v / total_raw
        else:
            # Fallback if everything is disabled (avoid div by zero)
            normalized_weights = {k: 0.0 for k in raw_weights}

        self.weights = TrustWeights(**normalized_weights)

    model_config = {"arbitrary_types_allowed": True}

# --- Constants ---

SOURCE_ID_FIELDS = [
    "source_id", "document_id", "file_id", "filename", 
    "url", "uri", "file_name", "source", "id", "doc_id"
]