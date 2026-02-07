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
- _Spearman Score_ checks whether more reliable answers consistently get higher scores than less reliable ones, regardless of the exact score values.

    - RELIABLE answers should score higher than ACCEPTABLE
    - ACCEPTABLE should score higher than UNRELIABLE

    If this ordering is respected most of the time, Spearman will be high.

- _Pearson Score_ checks whether moving up one trust level leads to a proportional increase in score.

    - UNRELIABLE → ACCEPTABLE should increase the score by about the same amount as
    - ACCEPTABLE → RELIABLE

    If score increases closely follow these step-by-step label increases, Pearson will be high.

**Results:**
```text
Spearman : 0.919 (The model almost always ranks answers correctly by trust level.)
Pearson  : 0.957 (The numerical scores increase in a very consistent, well-calibrated way as trust improves.)
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
