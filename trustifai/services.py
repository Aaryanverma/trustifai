# services.py
"""
Service layer for External APIs (LLMs, Embeddings, Rerankers).
"""

from typing import Optional, List, Any
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from litellm import completion, embedding, rerank, acompletion, batch_completion, aembedding
import litellm
from dotenv import load_dotenv
from trustifai.config import Config
from IPython import get_ipython
import httpx
import requests
import logging

logger = logging.getLogger(__name__)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except Exception as e:
    MLFLOW_AVAILABLE = False
    mlflow = None

litellm.drop_params = True

RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    OSError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    requests.Timeout,
    litellm.exceptions.Timeout,
    litellm.exceptions.APIConnectionError,
)


def is_notebook() -> bool:
    """Check if running in a Jupyter notebook environment"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (likely standard Python interpreter)
    except NameError:
        return False  # Probably standard Python interpreter


def empty_decorator(fn):
    return fn


TRACE_DECORATOR = (
    mlflow.trace
    if MLFLOW_AVAILABLE and getattr(mlflow, "trace", None)
    else empty_decorator
)


class ExternalService:
    def __init__(self, config: Config):
        self.config = config
        if self.config.env_file:
            load_dotenv(self.config.env_file)
            logger.info("Environment variables loaded.")
        self.configure_tracing()

    def configure_tracing(self):
        """Initialize tracing if enabled in config"""
        if not MLFLOW_AVAILABLE:
            return

        if self.config.tracing.params["enabled"]:
            mlflow.litellm.autolog(silent=True)
            mlflow.set_tracking_uri(self.config.tracing.params["tracking_uri"])

            experiment_name = self.config.tracing.params["experiment_name"]
            try:
                mlflow.set_experiment(experiment_name)
            except Exception:
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
        else:
            mlflow.litellm.autolog(disable=True, silent=True)

    @staticmethod
    @TRACE_DECORATOR
    def log_metrics_by_category(
        metrics_data: dict, trust_score: float, decision: str, offline_metric_keys: set
    ):
        """Log metrics to MLflow categorized by type"""
        if not MLFLOW_AVAILABLE:
            return

        # Categorize metrics
        offline_metrics = {}
        online_metrics = {}

        for key, value in metrics_data.items():
            metric_score = value["score"]
            if key == "source_diversity":
                metric_score = min(1.0, metric_score)

            if key in offline_metric_keys:
                key_name = f"offline/{key}"
                offline_metrics[key_name] = metric_score
            else:
                key_name = f"online/{key}"
                online_metrics[key_name] = metric_score

        # Log categorized metrics as attributes of the trace
        mlflow.set_tag("decision", decision)
        mlflow.set_tag("trust_score/final", trust_score)
        for k, v in offline_metrics.items():
            mlflow.set_tag(k, v)
        for k, v in online_metrics.items():
            mlflow.set_tag(k, v)

    @staticmethod
    def extract_document(document: Any) -> str:
        """Helper to extract text content from various document formats"""
        if document is None:
            return ""
        elif isinstance(document, list):
            if len(document) == 0:
                return ""
            elif len(document) == 1:
                document = document[0]
            else:
                return "\n".join(
                    [ExternalService.extract_document(doc) for doc in document]
                )
        elif isinstance(document, dict):
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

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=90),
        reraise=True,
    )
    def llm_call(
        self, system_prompt: str = None, prompt: str = None, **kwargs
    ) -> Optional[dict]:
        """Safely call LLM with retries using Config object"""
        system_prompt = system_prompt or "You are a helpful assistant."
        cfg = self.config.llm

        model = f"{cfg.type}/{cfg.params.get('model_name')}"
        base_url = cfg.params.get("base_url")
        api_base = cfg.params.get("api_base")
        api_version = cfg.params.get("api_version")
        deployment_id = cfg.params.get("deployment_id")
        final_kwargs = cfg.kwargs.copy()
        final_kwargs.update(kwargs)

        try:
            response = completion(
                model=model,
                base_url=base_url,
                api_base=api_base,
                api_version=api_version,
                deployment_id=deployment_id,
                seed=42,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **final_kwargs,
            )

            response_logprobs = None
            if (
                hasattr(response.choices[0], "logprobs")
                and response.choices[0].logprobs
            ):
                response_logprobs = [
                    token.logprob for token in response.choices[0].logprobs.content
                ]

            return {
                "response": response.choices[0].message.content,
                "logprobs": response_logprobs,
            }

        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise

        except Exception as e:
            logger.exception("Error calling LLM:", e)
            return {"response": None, "logprobs": None}

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=90),
        reraise=True,
    )
    def llm_call_batch(
        self, system_prompt: str = None, prompts: List[str] = None, **kwargs
    ) -> Optional[dict]:
        """Safely call LLM in batch with retries using Config object"""
        system_prompt = system_prompt or "You are a helpful assistant."
        cfg = self.config.llm

        model = f"{cfg.type}/{cfg.params.get('model_name')}"
        base_url = cfg.params.get("base_url", None)
        api_base = cfg.params.get("api_base")
        api_version = cfg.params.get("api_version")
        deployment_id = cfg.params.get("deployment_id")
        final_kwargs = cfg.kwargs.copy()
        final_kwargs.update(kwargs)

        try:
            response = batch_completion(
                model=model,
                base_url=base_url,
                api_base=api_base,
                api_version=api_version,
                deployment_id=deployment_id,
                seed=42,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *[{"role": "user", "content": prompt} for prompt in prompts],
                ],
                **final_kwargs,
            )

            response_logprobs = None
            if all(
                hasattr(choice, "logprobs") and choice.logprobs
                for choice in response.choices
            ):
                response_logprobs = [
                    [token.logprob for token in choice.logprobs.content]
                    for choice in response.choices
                ]

            return {
                "response": [choice.message.content for choice in response.choices],
                "logprobs": response_logprobs,
            }

        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise

        except Exception as e:
            logger.exception("Error calling LLM:", e)
            return {"response": None, "logprobs": None}

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=90),
        reraise=True,
    )
    async def llm_call_async(
        self, system_prompt: str = None, prompt: str = None, **kwargs
    ) -> Optional[dict]:
        """Safely call LLM asynchronously with retries using Config object"""
        system_prompt = system_prompt or "You are a helpful assistant."

        cfg = self.config.llm
        model = f"{cfg.type}/{cfg.params.get('model_name')}"
        base_url = cfg.params.get("base_url", None)
        api_base = cfg.params.get("api_base")
        api_version = cfg.params.get("api_version")
        deployment_id = cfg.params.get("deployment_id")
        final_kwargs = cfg.kwargs.copy()
        final_kwargs.update(kwargs)

        try:
            response = await acompletion(
                model=model,
                base_url=base_url,
                api_base=api_base,
                api_version=api_version,
                deployment_id=deployment_id,
                seed=42,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **final_kwargs,
            )

            response_logprobs = None
            if (
                hasattr(response.choices[0], "logprobs")
                and response.choices[0].logprobs
            ):
                response_logprobs = [
                    token.logprob for token in response.choices[0].logprobs.content
                ]

            return {
                "response": response.choices[0].message.content,
                "logprobs": response_logprobs,
            }

        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise

        except Exception as e:
            logger.exception("Error calling LLM asynchronously:", e)
            return {"response": None, "logprobs": None}

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=90),
        reraise=True,
    )
    def embedding_call(self, text: str, **kwargs) -> Optional[np.ndarray]:
        """Safely call embedding model"""
        cfg = self.config.embeddings
        model = f"{cfg.type}/{cfg.params.get('model_name')}"
        endpoint = cfg.params.get("endpoint", None)
        input_type = (
            "feature-extraction"
            if "huggingface" in model
            else kwargs.get("input_type", None)
        )

        try:
            response = embedding(
                model=model,
                input=[text],
                api_base=endpoint,
                input_type=input_type,
            )
            return response.data[0]["embedding"]
        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise
        except Exception as e:
            logger.exception(f"Error calling embedding: {e}")
            return None
        
    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=90),
        reraise=True,
    )
    async def embedding_call_async(self, text: str, **kwargs) -> Optional[np.ndarray]:
        """Safely call embedding model"""
        cfg = self.config.embeddings
        model = f"{cfg.type}/{cfg.params.get('model_name')}"
        endpoint = cfg.params.get("endpoint")
        input_type = (
            "feature-extraction"
            if "huggingface" in model
            else kwargs.get("input_type", None)
        )

        try:
            response = await aembedding(
                model=model,
                input=[text],
                api_base=endpoint,
                input_type=input_type,
            )
            return response.data[0]["embedding"]
        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise
        except Exception as e:
            logger.exception(f"Error calling embedding asynchronously: {e}")
            return None   

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=90),
        reraise=True,
    )
    def reranker_call(self, query: str, documents: List[str]):
        """Rerank documents based on similarity to query"""
        if not self.config.reranker or not self.config.reranker.type:
            logger.warning(
                "Warning: Reranker call attempted but no reranker configured."
            )
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
        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise
        except Exception as e:
            logger.exception(f"Error calling reranker: {e}")
            return []
