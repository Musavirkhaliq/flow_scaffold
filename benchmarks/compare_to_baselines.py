#!/usr/bin/env python3
"""
Compare our enhanced flow-scaffolding model against baseline benchmarks.

This script reads:
- Our metrics: results/enhanced_flow/analysis_*/metrics.json
- Baseline metrics: benchmarks/baselines/*.json

and prints side‑by‑side comparisons for key quantities:
- Diversity (mean pairwise distance)
- Ramachandran alpha/beta fractions
- Validity flags
- Any scalar metrics provided in the baseline JSON (e.g. TM-score, RMSD)
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def load_our_metrics(analysis_dir: Path) -> Dict[str, Any]:
    """Load our metrics.json from an analysis directory."""
    metrics_path = analysis_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json not found in {analysis_dir}")
    with metrics_path.open() as f:
        return json.load(f)


def load_baseline(name: str) -> Dict[str, Any]:
    """Load a baseline JSON by stem name from benchmarks/baselines/."""
    base_dir = Path(__file__).parent / "baselines"
    path = base_dir / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Baseline file not found: {path}")
    with path.open() as f:
        return json.load(f)


def summarize_our_metrics(our_metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert metrics.json (scenario -> metrics) into a flat DataFrame.

    Expected schema from train_and_evaluate_enhanced_flow.sh analysis script:
      {
        "scenario_name": {
          "angle_stats": {...},
          "ramachandran": {...},
          "diversity": {...},
          "validity": {...}
        },
        ...
      }
    """
    rows: List[Dict[str, Any]] = []
    for scenario, data in our_metrics.items():
        angle_stats = data.get("angle_stats", {})
        rama = data.get("ramachandran", {})
        div = data.get("diversity", {})
        validity = data.get("validity", {})

        row: Dict[str, Any] = {
            "Scenario": scenario,
            "n_samples": angle_stats.get("n_samples"),
            "seq_len": angle_stats.get("seq_len"),
            "mean_pairwise_diversity": div.get("mean_pairwise_distance"),
            "alpha_fraction": rama.get("alpha_fraction"),
            "beta_fraction": rama.get("beta_fraction"),
            "has_nan": validity.get("has_nan"),
            "has_inf": validity.get("has_inf"),
            "out_of_range_fraction": validity.get("out_of_range_fraction"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def compare_to_baseline(our_df: pd.DataFrame, baselines: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a simple textual comparison between our metrics and each baseline.

    Baseline JSON schema:
      {
        "name": "...",
        "metrics": {
          "tm_score_mean": ...,
          "rmsd_mean": ...,
          "success_rate": ...,
          "mean_pairwise_diversity": ...,
          "alpha_fraction": ...,
          "beta_fraction": ...
        }
      }
    """
    print("=" * 80)
    print("OUR MODEL VS BASELINES".center(80))
    print("=" * 80)
    print()

    print("Our scenarios and key metrics:\n")
    print(our_df.to_string(index=False))
    print("\n")

    for baseline_name, baseline in baselines.items():
        b_metrics = baseline.get("metrics", {})
        print("-" * 80)
        print(f"Baseline: {baseline.get('name', baseline_name)}")
        print(f"Source:   {baseline.get('source', 'N/A')}")
        print("-" * 80)

        # Build a small comparison table for diversity / rama fractions
        rows: List[Dict[str, Any]] = []
        for _, row in our_df.iterrows():
            rows.append(
                {
                    "Scenario": row["Scenario"],
                    "ours_diversity": row["mean_pairwise_diversity"],
                    "baseline_diversity": b_metrics.get("mean_pairwise_diversity"),
                    "ours_alpha": row["alpha_fraction"],
                    "baseline_alpha": b_metrics.get("alpha_fraction"),
                    "ours_beta": row["beta_fraction"],
                    "baseline_beta": b_metrics.get("beta_fraction"),
                }
            )

        cmp_df = pd.DataFrame(rows)
        print(cmp_df.to_string(index=False))
        print()

        # If TM-score / RMSD / success_rate are provided, print them too
        extra_keys = ["tm_score_mean", "tm_score_std", "rmsd_mean", "rmsd_std", "success_rate"]
        extra_rows = []
        for k in extra_keys:
            if k in b_metrics and b_metrics[k] is not None:
                extra_rows.append((k, b_metrics[k]))

        if extra_rows:
            print("Additional baseline-only metrics:")
            for k, v in extra_rows:
                print(f"  - {k}: {v}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare our enhanced flow-scaffolding model against baseline benchmarks."
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        required=True,
        help="Path to our analysis directory (e.g. results/enhanced_flow/analysis_enhanced_flow_*).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        action="append",
        default=[],
        help="Baseline name (stem of JSON in benchmarks/baselines/, e.g. 'foldingdiff_baseline'). "
        "Can be specified multiple times.",
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    our_metrics = load_our_metrics(analysis_dir)
    our_df = summarize_our_metrics(our_metrics)

    baselines: Dict[str, Dict[str, Any]] = {}
    for b in args.baseline:
        baselines[b] = load_baseline(b)

    if not baselines:
        print("No baselines specified. Use --baseline foldingdiff_baseline (for example).")
        print("Our metrics summary:\n")
        print(our_df.to_string(index=False))
        return

    compare_to_baseline(our_df, baselines)


if __name__ == "__main__":
    main()


