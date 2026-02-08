import pytest
from trustifai.metrics.calculators import CosineSimCalculator, ThresholdEvaluator, SourceIdentifier
from unittest.mock import MagicMock
import math

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

def test_cosine_similarity_nan():
    calc = CosineSimCalculator()
    sim = calc.calculate([0, 0], [0, 0])

    assert math.isnan(sim) or sim == 0.0

def test_source_identifier_priority(mock_service):
    """Test that different source IDs take precedence over content hash."""
    from trustifai.metrics.calculators import SourceIdentifier
    
    doc = MagicMock()
    
    # Case 1: 'source' in metadata
    doc.metadata = {"source": "src_1"}
    assert SourceIdentifier.resolve_source_id(doc, mock_service) == "source:src_1"
    
    # Case 2: 'file_name' in metadata (when source is missing)
    doc.metadata = {"file_name": "file.txt"}
    assert SourceIdentifier.resolve_source_id(doc, mock_service) == "file_name:file.txt"
    
    # Case 3: 'url' in metadata
    doc.metadata = {"url": "http://example.com"}
    assert SourceIdentifier.resolve_source_id(doc, mock_service) == "url:http://example.com"

def test_evaluator_consistency_levels():
    """Test all branches of consistency evaluation."""
    from trustifai.metrics.calculators import ThresholdEvaluator
    
    mock_config = MagicMock()
    mock_config.thresholds.STABLE_CONSISTENCY = 0.9
    mock_config.thresholds.FRAGILE_CONSISTENCY = 0.6
    
    evaluator = ThresholdEvaluator(mock_config)
    
    # Stable
    lbl, _ = evaluator.evaluate_consistency(0.95)
    assert lbl == "Stable Consistency"
    
    # Fragile
    lbl, _ = evaluator.evaluate_consistency(0.7)
    assert lbl == "Fragile Consistency"
    
    # Unreliable
    lbl, _ = evaluator.evaluate_consistency(0.5)
    assert lbl == "Unreliable"

def test_evaluator_confidence_levels():
    """Test all branches of confidence evaluation."""
    from trustifai.metrics.calculators import ThresholdEvaluator
    
    mock_config = MagicMock()
    mock_config.thresholds.HIGH_CONFIDENCE = 0.8
    mock_config.thresholds.MEDIUM_CONFIDENCE = 0.5
    
    evaluator = ThresholdEvaluator(mock_config)
    
    assert evaluator.evaluate_confidence(0.9)[0] == "High Confidence"
    assert evaluator.evaluate_confidence(0.6)[0] == "Medium Confidence"
    assert evaluator.evaluate_confidence(0.4)[0] == "Low Confidence"