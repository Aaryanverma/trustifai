# services.py
"""
Service layer for External APIs (LLMs, Embeddings, Rerankers).
"""

from typing import Optional, List, Any
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from litellm import completion, embedding, rerank
import litellm
from dotenv import load_dotenv
from trustifai.config import Config

litellm.drop_params = True

class ExternalService:
    def __init__(self, config: Config):
        self.config = config
        if self.config.env_file:
            load_dotenv(self.config.env_file)
            print("Environment variables loaded.")

    @staticmethod
    def extract_document(document: Any) -> str:
        """Helper to extract text content from various document formats"""
        if document is None:
            return ""
        if isinstance(document, list):
            document = document[0]
        if isinstance(document, dict):
            # Try common keys
            for key in ["page_content", "text", "content", "output", "document"]:
                if key in document:
                    return str(document[key])
        elif hasattr(document, "page_content"):
            return document.page_content
        elif hasattr(document, "text"):
            return document.text
        elif hasattr(document, "content"):
            return document.content
        elif hasattr(document, "output"):
            return document.output
        elif hasattr(document, "document"):
            return document.document
        return str(document)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=3, max=90))
    def llm_call(self, system_prompt: str = None, prompt: str = None, **kwargs) -> Optional[dict]:
        """Safely call LLM with retries using Config object"""
        system_prompt = system_prompt or "You are a helpful assistant."
        
        cfg = self.config.llm
        model = f"{cfg.type}/{cfg.params.get('model_name')}"
        endpoint = cfg.params.get("endpoint")
        
        # Merge config kwargs with runtime kwargs
        final_kwargs = cfg.kwargs.copy()
        final_kwargs.update(kwargs)

        try:
            response = completion(
                model=model,
                base_url=endpoint,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **final_kwargs,
            )
            
            response_logprobs = None
            if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
                 response_logprobs = [token.logprob for token in response.choices[0].logprobs.content]

            return {
                "response": response.choices[0].message.content,
                "logprobs": response_logprobs,
            }

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return {"response": None, "logprobs": None}

    def embedding_call(self, text: str) -> Optional[np.ndarray]:
        """Safely call embedding model"""
        cfg = self.config.embeddings
        model = f"{cfg.type}/{cfg.params.get('model_name')}"
        endpoint = cfg.params.get("endpoint")
        input_type = "feature-extraction" if "huggingface" in model else None
        
        try:
            response = embedding(
                model=model,
                input=[text],
                base_url=endpoint,
                input_type=input_type,
            )
            return response.data[0]["embedding"]
        except Exception as e:
            print(f"Error calling embedding: {e}")
            return []
        
    def reranker_call(self, query: str, documents: List[str]):
        """Rerank documents based on similarity to query"""
        if not self.config.reranker or not self.config.reranker.type:
            print("Warning: Reranker call attempted but no reranker configured.")
            return []

        cfg = self.config.reranker
        model = f"{cfg.type}/{cfg.params.get('model_name')}"
        top_n = cfg.params.get("top_n", len(documents))
        
        try:
            response = rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=top_n,
            )
            return response.results
        except Exception as e:
            print(f"Error calling reranker: {e}")
            return []