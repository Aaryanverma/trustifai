import pytest
from unittest.mock import patch, MagicMock
from trustifai.config import Config
from trustifai.services import ExternalService
from langchain_core.documents import Document

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
    import shutil
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

# --- Service Tests ---

def test_document_extraction():
    svc = ExternalService(MagicMock())
    
    # String
    assert svc.extract_document("Hello") == "Hello"
    # Dict
    assert svc.extract_document({"text": "Hello"}) == "Hello"
    # Object
    doc = Document(page_content="Hello")
    assert svc.extract_document(doc) == "Hello"
    # None
    assert svc.extract_document(None) == ""

@patch("trustifai.services.completion")
def test_llm_call_success(mock_completion):
    # Mock LiteLLM response structure
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Test Response"
    mock_response.choices[0].logprobs.content = []
    mock_completion.return_value = mock_response
    
    svc = ExternalService(MagicMock())
    svc.config.llm.type = "openai"
    
    res = svc.llm_call(prompt="Hi")
    assert res["response"] == "Test Response"

@patch("trustifai.services.completion")
def test_llm_call_failure(mock_completion):
    mock_completion.side_effect = Exception("API Error")
    svc = ExternalService(MagicMock())
    svc.config.llm.type = "openai"
    
    res = svc.llm_call(prompt="Hi")
    assert res["response"] is None