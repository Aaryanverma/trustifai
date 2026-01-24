# structures.py
"""
Shared data structures and types.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np
from pydantic import BaseModel, Field

# --- Enums ---

class TrustLevel(Enum):
    """Enumeration for trust levels across metrics"""
    STRONG = "Strong"
    PARTIAL = "Partial"
    WEAK = "Weak"
    RELIABLE = "Reliable"
    ACCEPTABLE = "Acceptable"
    UNRELIABLE = "Unreliable"
    HIGH = "High Trust"
    MODERATE = "Moderate Trust"
    LOW = "Low Trust"
    STABLE = "Stable Consistency"
    FRAGILE = "Fragile Consistency"
    NA = "N/A"

# --- Pydantic Models (Validation) ---

class SpanItem(BaseModel):
    index: int = Field(description="Index of the span")
    supported: bool = Field(description="True or False")
    answer: str = Field(description="answer span")

class SpanSchema(BaseModel):
    spans: List[SpanItem]

# --- Data Classes ---

@dataclass
class MetricContext:
    query: str
    answer: str
    documents: list
    query_embeddings: np.ndarray = None
    answer_embeddings: np.ndarray = None
    document_embeddings: np.ndarray = None

@dataclass
class MetricResult:
    """Standardized result structure for all metrics"""
    score: float
    label: str
    details: Dict
    
    def to_dict(self) -> Dict:
        return {
            "score": round(self.score, 2),
            "label": self.label,
            "details": self.details
        }

@dataclass
class SpanCheckResult:
    supported_count: int
    unsupported_spans: List[str]
    failed_count: int
    fail_reason: Optional[str]
    total_count: int

@dataclass
class RerankerResult:
    mean_score: float
    global_pass: bool
    fully_supported: int
    partially_supported: List
    detailed_results: List[Dict]

# --- Graph Structures ---

@dataclass
class ReasoningNode:
    node_id: str
    node_type: str 
    name: str
    inputs: Dict
    outputs: Dict
    score: Optional[float] = None
    label: Optional[str] = None
    details: Optional[Dict] = None

@dataclass
class ReasoningEdge:
    source: str
    target: str
    relationship: str 

@dataclass
class ReasoningGraph:
    trace_id: str
    nodes: List[ReasoningNode]
    edges: List[ReasoningEdge]

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "nodes": [node.__dict__ for node in self.nodes],
            "edges": [edge.__dict__ for edge in self.edges],
        }