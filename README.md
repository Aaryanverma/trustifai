# TrustifAI 
**🛡️Quantify, Visualize, and Explain Trust in AI.**

TrustifAI is a Python-based observability engine designed to evaluate the trustworthiness of LLM responses and Retrieval-Augmented Generation (RAG) systems. Unlike simple evaluation frameworks that rely on a single "correctness" score, TrustifAI computes a multi-dimensional **Trust Score** based on grounding, consistency, alignment, and diversity.

It also includes **visualizations** to help showcase why a model output was deemed unreliable.

![Build Status](https://github.com/aaryanverma/trustifai/actions/workflows/run-tests.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/trustifai.svg?icon=si%3Apython)](https://badge.fury.io/py/trustifai)

## 📊 Key Metrics

**TrustifAI** evaluates trustworthiness using four orthogonal vectors. The final *Trust Score* is a weighted aggregation of these components.

### Offline Metrics

| Metric | Definition | Purpose |
|------|------------|---------|
| **Evidence Coverage** | Segment-level entailment check. The answer is tokenized into sentences and each sentence is verified against retrieved documents using an NLI (Natural Language Inference) approach. | Detects hallucinations. Ensures every claim is supported by the provided context. |
| **Epistemic Consistency** | Measures semantic stability ($1 - \sigma$) across $k$ stochastic generations. Samples $k$ responses at high temperature and computes the mean cosine similarity against the original answer. | Detects model uncertainty. Hallucinated answers tend to vary significantly between runs. |
| **Semantic Drift** | Similarity between the Answer Embedding and the Mean Document Embedding. | Detects topic drift. Ensures the answer stays within the semantic envelope of the context. |
| **Source Diversity** | Normalized count of distinct source_id references contributing to the answer, adjusted using an exponential decay penalty. | Measures reliance on a single source while rewarding synthesis across multiple independent sources, without excessively penalizing cases where a single document is sufficient.

### Online Metrics

| Metric | Definition | Purpose |
|------|------------|---------|
| Confidence Score | Calculated using the log probabilities (logprobs) of the generated tokens. It considers the geometric mean of probabilities penalized by the variance of the generation. | Provides a real-time confidence signal (0.0−1.0) indicating how sure the model is about its own output.
 
## 🚀 Installation

TrustifAI requires Python 3.10+.

```python
pip install trustifai

#to enable tracing
pip install trustifai[trace]

# for tests
pip install trustifai[test]
```

OR

```
# Clone the repository
git clone https://github.com/Aaryanverma/trustifai.git
cd trustifai

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup
Create a .env file or export your API keys. TrustifAI uses LiteLLM, so it supports OpenAI, Azure, Anthropic, Gemini, Mistral, and more. (check .env.example)

## ⚡ Quick Start

**1. Evaluate an existing RAG Response in a few lines of code.**

  `Use this flow to score a query/answer pair against retrieved documents.`

```python
from trustifai import Trustifai, MetricContext
from langchain_core.documents import Document

# 1. Define your RAG Context
context = MetricContext(
    query="What is the capital of India?",
    answer="The capital is New Delhi.",
    documents=[
        Document(page_content="New Delhi is the capital of India.", metadata={"source": "wiki.txt"})
    ]
)

# 2. Initialize Engine
trust_engine = Trustifai(context)

# 3. Calculate Score
result = trust_engine.get_trust_score()
print(f"Trust Score: {result['score']} | Decision: {result['label']}")

# 4. Visualize Logic
graph = trust_engine.build_reasoning_graph(result)
trust_engine.visualize(graph, graph_type="pyvis") # Saves to reasoning_graph.html
```
![alt text](assets/trust_score_snippet.png)

**2. Generate with Confidence**

  `Use TrustifAI to generate a response and immediately get a confidence score based on token log probabilities.`

```python
from trustifai import Trustifai

# Initialize (Context will be None for pure generation)
trust_engine = Trustifai()

# Generate response
result = trust_engine.generate(
    prompt="What is the capital of France?",
    system_prompt="You are a helpful assistant."
)

print(f"Response: {result['response']}")
print(f"Confidence: {result['metadata']['confidence_score']} ({result['metadata']['confidence_label']})")
```

![alt text](assets/generate_snippet.png)

## ⚙️ Configuration
Control the sensitivity of the evaluation using config_file.yaml.

```python
#custom config can be passed on using config_path
trust_engine = Trustifai(context, config_path="config_file.yaml")
```
Refer: [config_file.yaml](config_file.yaml)

```yaml
# 1. Model Configuration (via LiteLLM)
llm:
  type: "openai"
  params:
    model_name: "gpt-5"

# 2. Thresholds (Strictness)
metrics:
  - type: "evidence_coverage"
    params:
      STRONG_GROUNDING: 0.85 # Threshold for "Trusted" label
      PARTIAL_GROUNDING: 0.60
  - type: "consistency"
    params:
      STABLE_CONSISTENCY: 0.90 # Requires 0.9 cosine sim to be "Stable"

# 3. Weighted Aggregation
# Adjust these based on your business priority.
score_weights:
  - type: "evidence_coverage"
    params: { weight: 0.45 } # Highest priority on factual accuracy
  - type: "semantic_drift"
    params: { weight: 0.30 }
  - type: "consistency"
    params: { weight: 0.20 }
  - type: "source_diversity"
    params: { weight: 0.05 }
```


## 🕸️ Reasoning Graphs

TrustifAI doesn't just give you a number; it gives you a map. The Reasoning Graph is a directed acyclic graph (DAG) representing the evaluation logic.
- Nodes: Represent individual metrics (Green=High Trust, Red=Low Trust).
- Edges: Represent the flow of data into the final aggregation.
- Interactive: The generated HTML uses PyVis for physics-based interaction.

To generate a graph:
```python
# Generate interactive HTML
trust_engine.visualize(graph, graph_type="pyvis")
```
![reasoning graph](assets/graph_gif.gif)

```python
# Generate Mermaid syntax for markdown documentation
print(trust_engine.visualize(graph, graph_type="mermaid"))
```
![mermaid diagram](assets/image-1.png)

## 🧩 Extending TrustifAI (Custom Metrics)

You can plug in custom evaluation logic without modifying the core library.

- Inherit from BaseMetric and implement calculate().

- Register the metric with a unique key.

- Configure the weight in your YAML file.

*Example: Adding a "Temporal Consistency" Metric*
```python
from trustifai.metrics import BaseMetric
from trustifai.structures import MetricResult

# 1. Define Metric
class TemporalConsistencyMetric(BaseMetric):
    """Detects temporal hallucinations - when the answer references dates/times
    that don't match the retrieved documents."""
    def calculate(self) -> MetricResult:
        # Extract dates from answer and documents
        answer_dates = self._extract_dates(self.context.answer) #assuming extract_dates logic is already implemented
        doc_dates = set()
        for doc in self.context.documents:
            doc_dates.update(self._extract_dates(doc.page_content))
        
        if not answer_dates:
            return MetricResult(
                score=1.0,
                label="No Temporal Claims",
                details={"answer_dates": [], "doc_dates": list(doc_dates)}
            )
        
        # Check if answer dates are within document date ranges
        supported_dates = [d for d in answer_dates if d in doc_dates]
        unsupported_dates = [d for d in answer_dates if d not in doc_dates]
        
        score = len(supported_dates) / len(answer_dates) if answer_dates else 1.0
        
        if score >= 0.8:
            label = "Temporally Consistent"
        elif score >= 0.5:
            label = "Partial Temporal Issues"
        else:
            label = "Temporal Hallucination Detected"
        
        return MetricResult(
            score=score,
            label=label,
            details={
                "answer_dates": answer_dates,
                "supported_dates": supported_dates,
                "unsupported_dates": unsupported_dates,
                "doc_dates": list(doc_dates)
            }
        )

# 2. Register Metric
from Trustifai import Trustifai
Trustifai.register_metric("temporal_consistency", TemporalConsistencyMetric)

# 3. Use in Trust Engine (Make sure to add it to config.yaml score_weights!)
trust_engine = Trustifai(context, "config_file.yaml")
```

*Updated config.yaml:*
```yaml
score_weights:
  - type: "evidence_coverage"
    params: { weight: 0.4 }
  - type: "temporal_consistency"         # <--- Your new metric
    params: { weight: 0.1 }   # Weights must sum to ~1.0
  # ... other metrics ...
```

## 🛠️ Architecture
- Context Ingestion: The MetricContext object normalizes inputs (Strings, LangChain/LlamaIndex Documents, List, Dictionary etc.).
- Vectorization: Embeddings for Query, Answer, and Docs are computed in parallel (if not provided in input).
- Metric Execution:
    - Coverage: Uses a Cross-Encoder Reranker or LLM (default) to verify span support.
    - Consistency: Triggers $k$ asynchronous generation calls to measure semantic variance.
- Confidence: Analyzes token-level logprobs during generation along with variance penalty.
- Aggregation: A weighted sum calculates the raw score [0, 1].
- Decision Boundary: The raw score is mapped to RELIABLE, ACCEPTABLE, or UNRELIABLE based on defined thresholds.

## 🎯 Benchmarks
- [Amnesty QA](benchmarks/amnesty_qa/benchmark_report.md)
- HaluEval (In progress)

## TODO
- [ ] Improve Tracing
- [ ] Benchmark Testing (In Progress) 
- [ ] Support for GraphRAG
- [x] Batch Processing
