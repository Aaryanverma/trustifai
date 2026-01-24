from unittest.mock import MagicMock
from trustifai import Trustifai
from trustifai.metrics import BaseMetric
from trustifai.structures import MetricResult
import tempfile
import yaml
import pytest

# --- Custom Metric for Testing ---
class MockCustomMetric(BaseMetric):
    def calculate(self) -> MetricResult:
        return MetricResult(score=1.0, label="Custom", details={})

def test_engine_initialization(basic_context, sample_config_yaml, mock_service):
    engine = Trustifai(basic_context, sample_config_yaml)
    # Inject mock service to avoid real calls during initialization if any
    engine.service = mock_service 
    
    assert engine.context == basic_context
    assert engine.config is not None

def test_get_trust_score(basic_context, sample_config_yaml, mock_service):
    engine = Trustifai(basic_context, sample_config_yaml)
    engine.service = mock_service
    
    # Mock individual metric calculations
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(return_value=MetricResult(score=0.8, label="Good", details={}))
    
    result = engine.get_trust_score()
    
    assert "score" in result
    assert "details" in result
    assert result["score"] > 0

def test_dynamic_metric_registration(basic_context, sample_config_yaml, mock_service):
    # 1. Register new metric
    Trustifai.register_metric("my_test_metric", MockCustomMetric)

    # 2. Load config and modify to include new metric
    with open(sample_config_yaml, 'r') as f:
        data = yaml.safe_load(f)
    data = dict(data)
    if 'score_weights' not in data:
        data['score_weights'] = []

    # Scale down existing weights to make room for the new metric
    new_weight = 0.5
    existing_weights = data['score_weights']
    total_existing = sum(item.get("params", {}).get("weight", 0.0) for item in existing_weights)
    if total_existing > 0:
        scale = (1.0 - new_weight) / total_existing
        for item in existing_weights:
            item["params"]["weight"] = item.get("params", {}).get("weight", 0.0) * scale

    data['score_weights'].append({"type": "my_test_metric", "params": {"weight": new_weight}})

    # 3. Write modified config to a temp file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp:
        yaml.dump(data, tmp)
        tmp.flush()
        tmp_path = tmp.name

    # 4. Init Engine with modified config path
    engine = Trustifai(basic_context, tmp_path)
    engine.service = mock_service

    # 5. Verify it exists in engine.metrics
    assert "my_test_metric" in engine.metrics
    assert isinstance(engine.metrics["my_test_metric"], MockCustomMetric)

    # 6. Verify it impacts score
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(return_value=MetricResult(score=1.0 if name == "my_test_metric" else 0.5, label="Good", details={}))
    result = engine.get_trust_score()
    assert result["score"] > 0.5  # Since my_test_metric returns 1.0 with significant weight

    # Clean up: Unregister the metric to avoid side effects on other tests
    if hasattr(Trustifai, "_metric_registry") and "my_test_metric" in Trustifai._metric_registry:
        Trustifai._metric_registry.pop("my_test_metric")

def test_generate_flow(sample_config_yaml, mock_service):
    engine = Trustifai(context=None, config_path=sample_config_yaml)
    engine.service = mock_service
    
    mock_service.llm_call.return_value = {
        "response": "Generated Answer",
        "logprobs": [-0.5, -0.5]
    }
    
    result = engine.generate("Prompt")
    assert result["response"] == "Generated Answer"
    assert "confidence_score" in result["metadata"]

def test_build_reasoning_graph(basic_context, sample_config_yaml, mock_service):
    engine = Trustifai(basic_context, sample_config_yaml)
    engine.service = mock_service
    
    # Mock metrics
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(return_value=MetricResult(score=0.5, label="OK", details={}))

    score = engine.get_trust_score()
    graph = engine.build_reasoning_graph(score)
    
    assert graph is not None
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    
def test_all_zero_weights_raises_error(basic_context, sample_config_yaml, mock_service):
    # Modify config so all weights are zero
    with open(sample_config_yaml, 'r') as f:
        data = yaml.safe_load(f)

    if 'score_weights' in data:
        for item in data['score_weights']:
            item['params']['weight'] = 0.0

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp:
        yaml.dump(data, tmp)
        tmp.flush()
        tmp_path = tmp.name

    engine = Trustifai(basic_context, tmp_path)
    engine.service = mock_service

    # Mock metrics
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(return_value=MetricResult(score=0.5, label="OK", details={}))

    # Should raise an error due to all-zero weights
    with pytest.raises(ValueError, match="all weights are zero"):
        engine.get_trust_score()

def test_generate_error_handling(sample_config_yaml, mock_service):
    engine = Trustifai(None, sample_config_yaml)
    engine.service = mock_service
    
    # Force failure
    mock_service.llm_call.return_value = None
    
    result = engine.generate("test")
    assert result['metadata']['error'] == "LLM call failed"