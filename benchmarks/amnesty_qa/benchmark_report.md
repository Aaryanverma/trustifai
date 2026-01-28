# TrustifAI Benchmark Report

**Generated on:** 2026-01-28 22:08:19

## Dataset Details

This benchmark is conducted using the [vibrantlabsai/amnesty_qa dataset](https://huggingface.co/datasets/vibrantlabsai/amnesty_qa) from huggingface which contains question-answer pairs related to human rights and Amnesty International reports. The dataset includes:

- 20 ground-truth answers sourced directly from verified Amnesty International documents
- 20 LLM-generated answers produced by querying language models
- Total of 40 QA pairs evaluated

The ground truth answers serve as a reliable baseline, while the LLM answers help assess TrustifAI's ability to detect potential hallucinations and inaccuracies in model-generated content.


## What Is Being Evaluated?

TrustifAI assigns a **trust score between 0 and 1** to each answer.

- **High score** → Reliable Answer
- **Moderate Score** → Acceptable answer (with caution)
- **Low score** → Unreliable (Likely Hallucinated) Answer

We evaluate TrustifAI on:
1. **LLM-generated answers**
2. **Ground-truth answers** (known to be correct)

**Expected behavior:** Ground-truth answers should consistently receive higher trust scores than LLM answers.


## Hallucination Detection (Binary Classification)

Labels are mapped as:
- **Trustworthy (1)** → RELIABLE, ACCEPTABLE (WITH CAUTION)
- **Untrustworthy (0)** → UNRELIABLE

**Interpretation:**
- ROC-AUC → separability between trustworthy vs untrustworthy answers
- PR-AUC → robustness under class imbalance

**Results:**
```text
ROC-AUC  : 1.000
PR-AUC   : 1.000
```

## Score Calibration (Ordinal Consistency)

Ordinal labels:
- UNRELIABLE = 0
- ACCEPTABLE (WITH CAUTION) = 1
- RELIABLE = 2

**Interpretation:**
- Spearman → Monotonic ordering:
 If answers labeled RELIABLE always score higher than ACCEPTABLE, and those score higher than UNRELIABLE, Spearman will be high.
- Pearson → Linear calibration strength:
 A one-step increase in label (e.g., UNRELIABLE → ACCEPTABLE) should correspond to a proportional increase in score.

**Results:**
```text
Spearman : 0.919
Pearson  : 0.957
```

## Reliability Distribution Comparison

A healthy system should assign:
- More **RELIABLE** labels to **Ground Truth**
- More **UNRELIABLE** labels to **LLM answers**


**Results:**

```text
Label         ACCEPTABLE (WITH CAUTION)  RELIABLE  UNRELIABLE
Type                                                         
Ground Truth                          0        19           1
LLM                                   6        10           4
```

## Verdict

TrustifAI demonstrates **meaningful separation** between grounded and hallucinated answers. Ground-truth responses consistently receive higher trust scores, indicating:

- Effective hallucination detection
- Reasonable score calibration
- Practical usefulness in RAG evaluation pipelines
