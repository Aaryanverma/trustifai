import numpy as np
from unittest.mock import MagicMock
from trustifai.metrics import (
    EvidenceCoverageMetric, 
    SemanticDriftMetric, 
    EpistemicConsistencyMetric,
    SourceDiversityMetric,
    ConfidenceMetric
)

def test_semantic_drift(basic_context, mock_service):
    # Setup embeddings to be identical
    basic_context.answer_embeddings = np.array([1, 0])
    basic_context.document_embeddings = np.array([[1, 0], [1, 0]])

    # Patch config with real thresholds
    config = MagicMock()
    config.thresholds = MagicMock()
    config.thresholds.STRONG_ALIGNMENT = 0.9
    config.thresholds.MODERATE_ALIGNMENT = 0.7
    config.thresholds.WEAK_ALIGNMENT = 0.5

    metric = SemanticDriftMetric(basic_context, mock_service, config)
    result = metric.calculate()

    assert result.score > 0.99
    assert result.label == "Strong Alignment"

def test_consistency(basic_context, mock_service):
    config = MagicMock()
    config.k_samples = 2
    config.thresholds = MagicMock()
    config.thresholds.STABLE_CONSISTENCY = 0.9
    config.thresholds.FRAGILE_CONSISTENCY = 0.7
    
    # Mock 2 generated samples identical to answer
    mock_service.llm_call_async.return_value = {"response": basic_context.answer, "logprobs": []}
    # Mock embeddings to return unit vectors
    mock_service.embedding_call.return_value = [1, 0]
    
    metric = EpistemicConsistencyMetric(basic_context, mock_service, config)
    result = metric.calculate()
    
    assert result.score > 0.99
    assert "Stable" in result.label

def test_source_diversity(basic_context, mock_service):
    # Context has 2 docs with different sources (wiki, geo_db)
    config = MagicMock()
    config.thresholds = MagicMock()
    config.thresholds.HIGH_DIVERSITY = 0.7
    config.thresholds.MODERATE_DIVERSITY = 0.4
    metric = SourceDiversityMetric(basic_context, mock_service, config)
    result = metric.calculate()
    
    assert result.details['unique_sources'] == 2
    assert result.score > 0.0

def test_evidence_coverage_llm(basic_context, mock_service):
    config = MagicMock()
    # Mock config to select LLM strategy
    config.metrics = [MagicMock(type="evidence_coverage", params={"strategy": "llm"})]
    config.reranker = None
    config.thresholds = MagicMock()
    config.thresholds.STRONG_GROUNDING = 0.9
    config.thresholds.PARTIAL_GROUNDING = 0.7
    
    # Mock LLM verification response
    # It expects a JSON response
    mock_service.llm_call.return_value = {
        "response": '{"spans": [{"index": 0, "supported": true, "answer": "span"}]}'
    }
    
    metric = EvidenceCoverageMetric(basic_context, mock_service, config)
    result = metric.calculate()
    
    assert result.score == 1.0 # 1 span, supported
    assert result.details['strategy'] == "LLM"

def test_confidence_metric():
    evaluator = MagicMock()
    evaluator.evaluate_confidence.return_value = ("High", "Explanation")

    # Perfect confidence (logprob 0 = prob 1)
    logprobs = [0.0, 0.0]
    result = ConfidenceMetric.calculate(logprobs, evaluator)

    # Use np.isclose for floating point comparison
    assert np.isclose(result['score'], 1.0)
    assert result['label'] == "High"

    # Empty logprobs
    result = ConfidenceMetric.calculate([], evaluator)
    assert result['label'] == "N/A"


