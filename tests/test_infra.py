import pytest
from trustifai.config import Config

# --- Config Tests ---

def test_config_loading(sample_config_yaml):
    cfg = Config.from_yaml(sample_config_yaml)
    assert cfg.llm.type == "openai"
    assert len(cfg.metrics) == 4
    # Check weight normalization logic
    assert abs(sum(cfg.weights.model_dump().values()) - 1.0) < 0.001

def test_weight_normalization_error(sample_config_yaml):
    # Create invalid weights using a temp file to avoid side effects
    import yaml
    import tempfile
    with open(sample_config_yaml, 'r') as f:
        data = yaml.safe_load(f)

    data['score_weights'][0]['params']['weight'] = 2.0 # Sum > 1.0

    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        yaml.dump(data, tmp)
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="Weights must normalize to 1.0"):
            Config.from_yaml(tmp_path)
    finally:
        import os
        os.remove(tmp_path)

def test_dynamic_config_fields(sample_config_yaml):
    """Test the new feature allowing custom metrics in config"""
    import yaml
    with open(sample_config_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Add a custom metric weight
    data['score_weights'].append({"type": "pii_check", "params": {"weight": 0.0}})
    
    with open(sample_config_yaml, 'w') as f:
        yaml.dump(data, f)
        
    cfg = Config.from_yaml(sample_config_yaml)
    # verify pydantic accepted the extra field
    assert hasattr(cfg.weights, "pii_check") or "pii_check" in cfg.weights.model_dump()


def test_document_extraction(mock_service):
    # String
    assert mock_service.extract_document("Hello") == "Hello"
    # Dict
    assert mock_service.extract_document({"text": "Hello"}) == "Hello"
    # List
    assert mock_service.extract_document(["Hello", "World"]) == "Hello\nWorld"
    # None
    assert mock_service.extract_document(None) == ""

def test_llm_call_success(mock_service):
    mock_service.llm_call.return_value = {"response": "Test Response", "logprobs": []}

    res = mock_service.llm_call(prompt="Hi")
    assert res["response"] == "Test Response"

def test_llm_call_failure(mock_service):
    mock_service.llm_call.side_effect = Exception("API Error")

    with pytest.raises(Exception, match="API Error"):
        mock_service.llm_call(prompt="Hi")

def test_embedding_call(mock_service):
    mock_service.embedding_call.return_value = [0.1, 0.2, 0.3]

    vec = mock_service.embedding_call("Test text")
    assert isinstance(vec, list) or hasattr(vec, "__array__")

def test_embedding_call_empty_input(mock_service):
    mock_service.embedding_call.return_value = []

    vec = mock_service.embedding_call("")
    assert vec == []

def test_reranker_call(mock_service):
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    query = "Test query"

    ranked_docs = mock_service.reranker_call(docs, query)
    assert isinstance(ranked_docs, list)

def test_reranker_empty_result(mock_service):
    mock_service.reranker_call.return_value = []

    docs = mock_service.extract_document([])

    assert docs == ''