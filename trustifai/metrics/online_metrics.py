# online_metrics.py
"""
Online metrics calculated during LLM generation time.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from trustifai.metrics.calculators import ThresholdEvaluator
from trustifai.visualizer import ConfidenceVisualizer
from nltk.tokenize import sent_tokenize


class ConfidenceMetric:
    """
    Calculates the confidence of the LLM response using log probabilities
    captured during the generation process.
    """

    @staticmethod
    def calculate(
        logprobs: List[Any], evaluator: ThresholdEvaluator, visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Computes the confidence score from a list of token log probability objects.

        Args:
            logprobs: List of token objects (e.g. response.choices[0].logprobs.content).
                      Each item must have .token and .logprob attributes (or be a dict).
            evaluator: ThresholdEvaluator instance to classify the score.

        Returns:
            Dict containing score, label, and detailed explanation with HTML visualization.
        """
        if not logprobs:
            return {
                "score": 0.0,
                "label": "N/A",
                "details": {
                    "explanation": "No logprobs available for confidence calculation."
                },
            }

        try:
            # extract (token, logprob) tuples from input objects
            extracted_data: List[Tuple[str, float]] = []

            for item in logprobs:
                if isinstance(item, dict):
                    t = item.get("token")
                    lp = item.get("logprob")
                else:
                    t = getattr(item, "token", None)
                    lp = getattr(item, "logprob", None)

                # Only valid if we have both text and a score
                if t is not None and lp is not None:
                    extracted_data.append((t, float(lp)))

            if not extracted_data:
                return {
                    "score": 0.0,
                    "label": "Error",
                    "details": {
                        "explanation": "Invalid input: Could not find 'token' and 'logprob' attributes in the provided list."
                    },
                }

            # Unzip for vectorized numpy calculations
            tokens_list, logprob_list = zip(*extracted_data)
            logprob_array = np.array(logprob_list)

            # Global Score Calculation (Geometric Mean * Variance Penalty)
            avg_logprob = np.mean(logprob_array)
            variance = np.var(logprob_array)

            # Geometric mean: exp(mean(logprobs)) maps (-inf, 0] to [0, 1]
            logprob_score = float(np.exp(avg_logprob))

            # Penalty for high variance (indicating uncertainty/inconsistency)
            penalty = np.exp(-variance)

            final_score = round(logprob_score * penalty, 2)
            label, explanation = evaluator.evaluate_confidence(final_score)

            # Pass the clean extracted data to helper
            sentence_details = ConfidenceMetric._calculate_sentence_confidence(
                extracted_data, evaluator
            )
            details = {
                "explanation": explanation,
                "avg_logprob": round(avg_logprob, 2),
                "variance": round(variance, 2),
                "token_count": len(extracted_data),
                "sentence_analysis": sentence_details,
            }
            
            if visualize:
                html_viz = ConfidenceVisualizer._generate_html(sentence_details)
                viz = {"html": html_viz}
                details.update(**viz)

            return {
                "score": final_score,
                "label": label,
                "details": details
            }

        except Exception as e:
            return {
                "score": 0.0,
                "label": "Error",
                "details": {"explanation": f"Calculation error: {str(e)}"},
            }

    @staticmethod
    def _calculate_sentence_confidence(
        extracted_data: List[Tuple[str, float]], evaluator: ThresholdEvaluator
    ) -> List[Dict]:
        """
        Groups token logprobs into sentences and calculates confidence.
        """
        full_text = "".join([t for t, lp in extracted_data])

        try:
            sent_texts = sent_tokenize(full_text)
        except Exception:
            sent_texts = [full_text]

        sentences = []
        token_idx = 0
        total_tokens = len(extracted_data)

        for text in sent_texts:
            target_len = len(text)
            current_len = 0
            sent_probs = []
            current_sent_text = ""

            while current_len < target_len and token_idx < total_tokens:
                token_str, lp = extracted_data[token_idx]

                # Convert logprob to linear probability [0, 1]
                sent_probs.append(np.exp(lp))
                current_sent_text += token_str
                current_len += len(token_str)
                token_idx += 1

            # Geometric mean for this specific sentence
            if sent_probs:
                # Filter out exact 0.0 to prevent log(0) errors (rare but possible)
                safe_probs = [p for p in sent_probs if p > 0]
                if safe_probs:
                    geo_mean = float(np.exp(np.mean(np.log(safe_probs))))
                else:
                    geo_mean = 0.0
            else:
                geo_mean = 0.0

            label, _ = evaluator.evaluate_confidence(geo_mean)

            sentences.append(
                {
                    "text": current_sent_text,
                    "score": round(geo_mean, 2),
                    "label": label,
                    "token_count": len(sent_probs),
                }
            )

        return sentences

    