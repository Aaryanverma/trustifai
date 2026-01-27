import sys

import numpy as np

sys.path.append("../.")
import os
import argparse
import asyncio
import logging
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

try:
    from trustifai import Trustifai, MetricContext
except ImportError:
    logger.error(
        "Could not import 'trustifai'. Ensure it is installed: pip install trustifai."
    )
    sys.exit(1)

# Constants
LABEL_MAP_ORDINAL = {
    "UNRELIABLE": 0,
    "ACCEPTABLE (WITH CAUTION)": 1,
    "RELIABLE": 2,
}

LABEL_MAP_BINARY = {
    "UNRELIABLE": 0,
    "ACCEPTABLE (WITH CAUTION)": 1,
    "RELIABLE": 1,
}


class TrustifAIBenchmark:
    """
    A modular benchmarking suite for evaluating TrustifAI performance.
    Handles data loading, async scoring, and metric calculation.
    """

    def __init__(self, config_path: str, concurrency: int = 5, timeout: int = 30):
        """
        Initialize the benchmark.

        Args:
            config_path: Path to the TrustifAI YAML config file.
            concurrency: Number of concurrent async tasks for scoring.
        """
        self.config_path = config_path
        self.sem = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.results_df: Optional[pd.DataFrame] = None

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

    def load_data(
        self, dataset_path: str, split: str = "eval", name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Loads the dataset from HuggingFace or a local file.
        """
        logger.info(f"Loading dataset: {dataset_path} (split={split})")
        try:
            if dataset_path.endswith(".json"):
                ds = load_dataset("json", data_files=dataset_path, split=split)
            else:
                ds = (
                    load_dataset(dataset_path, name, split=split)
                    if name
                    else load_dataset(dataset_path, split=split)
                )

            logger.info(f"Loaded {len(ds)} records.")
            return ds
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            sys.exit(1)

    def _score_row_sync(
        self,
        row: Dict[str, Any],
        query_column: str = "user_input",
        answer_column: str = "response",
        context_column: str = "retrieved_contexts",
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for the TrustifAI engine.
        """
        context = MetricContext(
            query=row.get(query_column),
            answer=row.get(answer_column) or row.get("reference"),
            documents=row.get(context_column, []),
        )

        engine = Trustifai(context, self.config_path)
        return engine.get_trust_score()

    async def _score_row_async(self, row: Dict[str, Any]) -> Tuple[float, str]:
        """
        Async wrapper to limit concurrency using a semaphore.
        """
        try:
            async with self.sem:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self._score_row_sync, row), timeout=self.timeout
                )
                return result["score"], result["label"]
        except asyncio.TimeoutError:
            logger.warning("Row processing timed out.")
            return np.nan, "ERROR"
        except Exception as e:
            logger.warning(f"Row processing failed: {e}")
            return np.nan, "ERROR"

    async def run_evaluation(self, dataset: Any) -> pd.DataFrame:
        """
        Runs the evaluation loop asynchronously.
        """
        logger.info("Starting evaluation...")
        tasks = [self._score_row_async(row) for row in dataset]

        results = await tqdm_asyncio.gather(*tasks)

        scores, labels = zip(*results)

        df = dataset.to_pandas()
        df["response_score"] = list(scores)
        df["response_label"] = list(labels)

        error_count = df[df["response_label"] == "ERROR"].shape[0]
        if error_count > 0:
            logger.warning(f"Dropped {error_count} rows due to timeouts or errors.")
            df = df[df["response_label"] != "ERROR"].copy()

        df = self._prepare_labels(df)
        self.results_df = df
        return df

    def _prepare_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps string labels to ordinal and binary numeric values.
        """
        df["response_label_ordinal"] = df["response_label"].map(LABEL_MAP_ORDINAL)
        df["response_label_binary"] = df["response_label"].map(LABEL_MAP_BINARY)

        if "ground_truth_label" in df.columns:
            df["ground_truth_label_ordinal"] = df["ground_truth_label"].map(
                LABEL_MAP_ORDINAL
            )
            df["ground_truth_label_binary"] = df["ground_truth_label"].map(
                LABEL_MAP_BINARY
            )

        return df

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Computes Task A (Binary) and Task B (Ordinal) metrics.
        """
        if self.results_df is None:
            raise ValueError("No results to compute. Run evaluate() first.")

        df = self.results_df
        metrics = {}

        # --- Task A: Binary Hallucination Detection ---
        # We treat RELIABLE and ACCEPTABLE as Positive (1), UNRELIABLE as Negative (0)
        try:
            if "response_label_binary" in df and "response_score" in df:
                clean_df = df.dropna(subset=["response_label_binary", "response_score"])
                if not clean_df.empty:
                    metrics["response_roc_auc"] = roc_auc_score(
                        clean_df["response_label_binary"], clean_df["response_score"]
                    )
                    metrics["response_pr_auc"] = average_precision_score(
                        clean_df["response_label_binary"], clean_df["response_score"]
                    )

            if "ground_truth_label_binary" in df and "ground_truth_score" in df:
                # If ground truth scores were computed (usually they are static labels, but logic copied from nb)
                pass
        except Exception as e:
            logger.warning(f"Could not compute binary metrics: {e}")

        # --- Task B: Ordinal Calibration ---
        # Correlation between continuous score and ordinal rank
        try:
            if "response_label_ordinal" in df:
                clean_df = df.dropna(
                    subset=["response_label_ordinal", "response_score"]
                )
                if not clean_df.empty:
                    metrics["spearman_corr"] = spearmanr(
                        clean_df["response_label_ordinal"], clean_df["response_score"]
                    ).correlation
                    metrics["pearson_corr"] = pearsonr(
                        clean_df["response_label_ordinal"], clean_df["response_score"]
                    )[0]
        except Exception as e:
            logger.warning(f"Could not compute correlation metrics: {e}")

        return metrics

    def generate_report(
        self, metrics: Dict[str, Any], export_path: Optional[str] = None
    ):
        """
        Generates a structured report of the findings and optionally saves it to a Markdown file.
        """
        lines = []
        lines.append("# TrustifAI Benchmark Report")
        lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        lines.append("## Task A: Binary Hallucination Detection")
        lines.append(f"- **ROC AUC:** {metrics.get('response_roc_auc', 'N/A'):.4f}")
        lines.append(f"- **PR AUC:** {metrics.get('response_pr_auc', 'N/A'):.4f}")
        lines.append("")

        lines.append("## Task B: Ordinal Calibration")
        lines.append(
            f"- **Spearman Correlation:** {metrics.get('spearman_corr', 'N/A'):.4f}"
        )
        lines.append(
            f"- **Pearson Correlation:** {metrics.get('pearson_corr', 'N/A'):.4f}"
        )
        lines.append("")

        lines.append("## Score Distribution by Label")
        if self.results_df is not None:
            stats = self.results_df.groupby("response_label")["response_score"].agg(
                ["mean", "std", "median", "min", "max", "count"]
            )
            lines.append("### Statistics")
            lines.append("```")
            lines.append(stats.to_string())
            lines.append("```")
            lines.append("")

            # Print Label Distribution Comparison (LLM vs Ground Truth) if available
            if "ground_truth_label" in self.results_df.columns:
                lines.append("### Label Distribution (Normalized)")
                long_df = pd.DataFrame(
                    {
                        "Answer Type": ["LLM Response"] * len(self.results_df)
                        + ["Ground Truth"] * len(self.results_df),
                        "Label": pd.concat(
                            [
                                self.results_df["response_label"],
                                self.results_df["ground_truth_label"],
                            ],
                            ignore_index=True,
                        ),
                    }
                )
                counts = (
                    long_df.groupby("Answer Type")["Label"]
                    .value_counts()
                    .unstack(fill_value=0)
                )
                percentages = counts.div(counts.sum(axis=1), axis=0).round(3)
                lines.append("```")
                lines.append(percentages.to_string())
                lines.append("```")
                lines.append("")

        report_content = "\n".join(lines)

        # Print to console
        print("\n" + "=" * 50)
        print("BENCHMARK REPORT")
        print("=" * 50)
        print(report_content)
        print("=" * 50 + "\n")

        # Save to file if path provided
        if export_path:
            try:
                with open(export_path, "w") as f:
                    f.write(report_content)
                logger.info(f"Report saved to {export_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")

    def save_results(self, output_path: str):
        """Saves the detailed DataFrame to CSV."""
        if self.results_df is not None:
            self.results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="TrustifAI Benchmark Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="../config_file.yaml",
        help="Path to TrustifAI config yaml",
    )
    parser.add_argument(
        "--env-file", type=str, default="../creds.env", help="Path to .env file"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="vibrantlabsai/amnesty_qa",
        help="HuggingFace dataset name or json path",
    )
    parser.add_argument(
        "--name", type=str, default="english_v3", help="Dataset subset name"
    )
    parser.add_argument("--split", type=str, default="eval", help="Dataset split")
    parser.add_argument(
        "--output", type=str, default="benchmark_results.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="benchmark_report.md",
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Async concurrency limit"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout in seconds per row"
    )

    args = parser.parse_args()

    # Load environment
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)
        logger.info(f"Loaded environment from {args.env_file}")
    else:
        logger.warning(
            f"Env file not found at {args.env_file}. Ensure env vars are set manually."
        )

    # Initialize Benchmark
    try:
        benchmark = TrustifAIBenchmark(
            config_path=args.config, concurrency=args.concurrency, timeout=args.timeout
        )

        # Load Data
        dataset = benchmark.load_data(
            args.dataset_path, split=args.split, name=args.name
        )

        # Run Evaluation
        await benchmark.run_evaluation(dataset)

        # Compute Metrics
        metrics = benchmark.compute_metrics()

        # Report & Save
        benchmark.generate_report(metrics)
        benchmark.save_results(args.output)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        # import traceback
        # traceback.print_exc()


if __name__ == "__main__":
    # Suppress specific Pydantic warnings if necessary
    warnings.filterwarnings("ignore", category=UserWarning)

    asyncio.run(main())
