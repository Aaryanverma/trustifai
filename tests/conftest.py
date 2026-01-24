import pytest
import yaml
import tempfile
import os
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from langchain_core.documents import Document
import sys
sys.path.append("./.")

from trustifai.structures import MetricContext
from trustifai.services import ExternalService

@pytest.fixture
def sample_config_yaml():
    """Creates a temporary config.yaml file for testing"""
    config_data = {
        "env_file": ".env.test",
        "tracing": {"type": "default", "params": {"enabled": False}},
        "llm": {"type": "openai", "params": {"model_name": "gpt-4"}},
        "embeddings": {"type": "openai", "params": {"model_name": "text-embedding-3-small"}},
        "reranker": {"type": "cohere", "params": {"model_name": "rerank-v3.5"}},
        "metrics": [
            {"type": "evidence_coverage", "enabled": True, "params": {"strategy": "llm"}},
            {"type": "semantic_drift", "enabled": True},
            {"type": "consistency", "enabled": True},
            {"type": "source_diversity", "enabled": True}
        ],
        "score_weights": [
            {"type": "evidence_coverage", "params": {"weight": 0.4}},
            {"type": "semantic_drift", "params": {"weight": 0.3}},
            {"type": "consistency", "params": {"weight": 0.2}},
            {"type": "source_diversity", "params": {"weight": 0.1}}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        path = f.name
        
    yield path
    os.remove(path)

@pytest.fixture
def mock_service():
    """Mocks the ExternalService to prevent real API calls"""
    service = MagicMock(spec=ExternalService)
    
    # Default behaviors
    service.llm_call = MagicMock()
    service.llm_call_async = AsyncMock()
    service.embedding_call = MagicMock()
    service.reranker_call = MagicMock()
    service.extract_document = ExternalService.extract_document
    service.embedding_call.return_value = np.array([0.1, 0.2, 0.3])
    service.llm_call.return_value = {"response": "Mocked LLM Response", "logprobs": [-0.1, -0.2]}
    service.llm_call_async.return_value = {"response": "Mocked Async LLM Response", "logprobs": [-0.1, -0.2]}
    service.reranker_call.return_value = ["Doc 1", "Doc 2", "Doc 3"]
    
    return service

@pytest.fixture
def basic_context():
    """Provides a standard MetricContext with dummy data"""
    docs = [
        Document(page_content="Delhi is the capital of India.", metadata={"source": "wiki"}),
        Document(page_content="India's capital is Delhi.", metadata={"source": "geo_db"})
    ]
    return MetricContext(
        query="What is the capital of India?",
        answer="The capital of India is Delhi.",
        documents=docs,
        query_embeddings=np.array([0.1, 0.2, 0.3]),
        answer_embeddings=np.array([0.1, 0.2, 0.3]),
        document_embeddings=[np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])]
    )