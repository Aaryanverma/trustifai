import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from trustifai.services import ExternalService, is_notebook
from trustifai.config import Config

# --- Robust Fixtures to fix AttributeError ---
@pytest.fixture
def mock_config():
    """
    Creates a mock configuration object that mimics the nested pydantic structure
    to prevent 'AttributeError: Mock object has no attribute ...'
    """
    config = MagicMock(spec=Config)
    
    # 1. Fix Tracing Config
    config.tracing = MagicMock()
    config.tracing.params = {"enabled": False, "tracking_uri": "http://localhost:5000", "experiment_name": "test"}
    
    # 2. Fix LLM Config
    config.llm = MagicMock()
    config.llm.type = "openai"
    config.llm.params = {"model_name": "gpt-4", "base_url": None}
    config.llm.kwargs = {}
    
    # 3. Fix Embeddings Config
    config.embeddings = MagicMock()
    config.embeddings.type = "openai"
    config.embeddings.params = {"model_name": "text-embedding-3-small"}
    
    # 4. Fix Reranker Config
    config.reranker = MagicMock()
    config.reranker.type = "cohere"
    config.reranker.params = {"model_name": "rerank-english"}
    
    config.env_file = None
    return config

@pytest.fixture
def service(mock_config):
    return ExternalService(mock_config)

# --- Test Document Extraction (High Complexity Area) ---
class TestDocumentExtraction:
    def test_extract_none_and_empty(self, service):
        assert service.extract_document(None) == ""
        assert service.extract_document([]) == ""
        assert service.extract_document({}) == "{}"

    def test_extract_nested_list(self, service):
        """Test recursive extraction of lists."""
        # Single item list
        assert service.extract_document(["hello"]) == "hello"
        # Multi-item list
        assert service.extract_document(["hello", "world"]) == "hello\nworld"
        # Nested list
        assert service.extract_document([["a"], "b"]) == "a\nb"

    def test_extract_dict_priorities(self, service):
        """Test dictionary key priority."""
        assert service.extract_document({"page_content": "content"}) == "content"
        assert service.extract_document({"text": "text_val"}) == "text_val"
        # Should convert unknown types to string
        assert service.extract_document({"unknown": 123}) == "{'unknown': 123}"

    def test_extract_object_attributes(self, service):
        """Test object attribute priority."""
        obj = MagicMock()
        obj.page_content = "obj_content"
        assert service.extract_document(obj) == "obj_content"
        
        del obj.page_content
        obj.text = "obj_text"
        assert service.extract_document(obj) == "obj_text"

# --- Test LLM Logic & Error Handling ---
class TestLLMCalls:
    def test_llm_call_with_image(self, service):
        """Test that image_url triggers the multimodal message structure."""
        with patch("trustifai.services.completion") as mock_comp:
            mock_comp.return_value.choices = [MagicMock(message=MagicMock(content="Image Desc"))]
            
            service.llm_call(prompt="Describe", image_url="http://img.com/a.png")
            
            # Verify structure passed to completion
            call_kwargs = mock_comp.call_args[1]
            messages = call_kwargs["messages"]
            assert len(messages) == 2
            # Check user content is list (multimodal)
            assert isinstance(messages[1]["content"], list)
            assert messages[1]["content"][1]["image_url"]["url"] == "http://img.com/a.png"

    def test_llm_call_batch_mixed_errors(self, service):
        """Test batch call where some inputs might fail or return no logprobs."""
        with patch("trustifai.services.batch_completion") as mock_batch:
            # Setup response 1: Success with logprobs
            r1 = MagicMock()
            r1.choices = [MagicMock(message=MagicMock(content="A"))]
            r1.choices[0].logprobs.content = [MagicMock(logprob=-0.1)]
            
            # Setup response 2: Success NO logprobs
            r2 = MagicMock()
            r2.choices = [MagicMock(message=MagicMock(content="B"))]
            r2.choices[0].logprobs = None 

            mock_batch.return_value = [r1, r2]
            
            result = service.llm_call_batch(prompts=["1", "2"])
            assert result["response"] == ["A", "B"]
            assert result["logprobs"][0] is not None
            assert result["logprobs"][1] is None

# --- Test Environment & Tracing ---
def test_is_notebook_detection():
    # 1. ZMQ (Jupyter)
    with patch("trustifai.services.get_ipython") as mock_ip:
        mock_ip.return_value.__class__.__name__ = "ZMQInteractiveShell"
        assert is_notebook() is True
    
    # 2. Terminal
    with patch("trustifai.services.get_ipython") as mock_ip:
        mock_ip.return_value.__class__.__name__ = "TerminalInteractiveShell"
        assert is_notebook() is False
        
    # 3. NameError (Standard Python)
    with patch("trustifai.services.get_ipython", side_effect=NameError):
        assert is_notebook() is False

def test_mlflow_logging_logic(service):
    """Test the categorization logic in log_metrics_by_category."""
    # Force MLFLOW_AVAILABLE to True for this test scope
    with patch("trustifai.services.MLFLOW_AVAILABLE", True):
        with patch("trustifai.services.mlflow") as mock_mlflow:
            metrics = {
                "grounding": {"score": 0.1},
                "confidence": {"score": 0.9}
            }
            # 'grounding' is offline, 'confidence' is online
            service.log_metrics_by_category(
                metrics, 0.5, "Unsure", offline_metric_keys={"grounding"}
            )
            
            tags = {c[0][0]: c[0][1] for c in mock_mlflow.set_tag.call_args_list}
            assert tags["offline/grounding"] == 0.1
            assert tags["online/confidence"] == 0.9
            assert tags["decision"] == "Unsure"