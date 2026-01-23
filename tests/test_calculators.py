import pytest
from trustifai.metrics.calculators import CosineSimCalculator, ThresholdEvaluator, SourceIdentifier
from unittest.mock import MagicMock

def test_cosine_similarity():
    calc = CosineSimCalculator()
    
    # Identical vectors
    v1 = [1, 0, 0]
    assert calc.calculate(v1, v1) > 0.99
    
    # Orthogonal vectors
    v2 = [0, 1, 0]
    assert calc.calculate(v1, v2) == 0.0
    
    # Zero vector handling
    v3 = [0, 0, 0]
    assert calc.calculate(v1, v3) == 0.0
    
    # Missing input
    with pytest.raises(ValueError):
        calc.calculate(None, v1)

def test_threshold_evaluator():
    mock_config = MagicMock()
    mock_config.thresholds.STRONG_GROUNDING = 0.8
    mock_config.thresholds.PARTIAL_GROUNDING = 0.5
    
    evaluator = ThresholdEvaluator(mock_config)
    
    # Grounding
    lbl, _ = evaluator.evaluate_grounding(0.9)
    assert lbl == "Strong Grounding"
    
    lbl, _ = evaluator.evaluate_grounding(0.6)
    assert lbl == "Partial Grounding"
    
    lbl, _ = evaluator.evaluate_grounding(0.2)
    assert "Hallucinated" in lbl


# Add a pytest fixture for mock_service
@pytest.fixture
def mock_service():
    return MagicMock()

def test_source_identifier(mock_service):
    doc = MagicMock()
    doc.metadata = {"source_id": "123"}

    sid = SourceIdentifier()
    res = sid.resolve_source_id(doc, mock_service)
    assert res == "source_id:123"

    # Fallback to hash
    doc.metadata = {}
    mock_service.extract_document.return_value = "content"
    res = sid.resolve_source_id(doc, mock_service)
    assert "content_hash" in res