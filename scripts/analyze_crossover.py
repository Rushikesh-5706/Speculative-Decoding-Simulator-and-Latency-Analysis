"""
analyze_crossover.py — Compute break-even acceptance rate and domain comparisons.

Reads results/sweep_metrics.csv, fits a linear regression of speedup ratio
on acceptance rate, solves for the crossover point, and writes crossover_report.json.
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def compute_crossover(csv_path: str, out_path: str) -> dict:
    df = pd.read_csv(csv_path)

    # Validate expected columns
    required = {"prompt_id", "domain", "method", "n_draft", "latency_sec", "tokens_per_sec", "acceptance_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sweep_metrics.csv is missing columns: {missing}")

    baseline_df = df[df["method"] == "baseline"]
    speculative_df = df[df["method"] == "speculative"]

    # Mean baseline tokens/sec per domain
    baseline_by_domain = (
        baseline_df.groupby("domain")["tokens_per_sec"].mean().to_dict()
    )
    logger.info("Baseline tokens/sec by domain: %s", baseline_by_domain)

    # Compute speedup ratio for every speculative row
    spec_rows = speculative_df.copy()
    spec_rows["baseline_tps"] = spec_rows["domain"].map(baseline_by_domain)
    spec_rows["speedup_ratio"] = spec_rows["tokens_per_sec"] / spec_rows["baseline_tps"]

    # Determine slower and faster domain
    spec_mean_by_domain = spec_rows.groupby("domain")["tokens_per_sec"].mean()
    domain_names = list(spec_mean_by_domain.index)

    if len(domain_names) >= 2:
        slower = spec_mean_by_domain.idxmin()
        faster = spec_mean_by_domain.idxmax()
    elif len(domain_names) == 1:
        slower = domain_names[0]
        faster = domain_names[0]
        logger.warning("Only one domain found in speculative rows — slower/faster are the same.")
    else:
        slower = "unknown"
        faster = "unknown"
        logger.warning("No speculative rows found.")

    # Optimal n_draft: highest mean tokens/sec across both domains
    spec_by_ndraft = spec_rows.groupby("n_draft")["tokens_per_sec"].mean()
    optimal_n_draft = int(spec_by_ndraft.idxmax()) if not spec_by_ndraft.empty else 0

    # Break-even acceptance rate via linear regression
    ar_vals = spec_rows["acceptance_rate"].values
    sr_vals = spec_rows["speedup_ratio"].values

    break_even = _compute_break_even(ar_vals, sr_vals)

    result = {
        "break_even_acceptance_rate": break_even,
        "slower_domain_name": slower,
        "faster_domain_name": faster,
        "optimal_n_draft": optimal_n_draft,
    }

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("crossover_report.json written to %s", out_path)
    return result


def _compute_break_even(ar_vals: np.ndarray, sr_vals: np.ndarray) -> float:
    """
    Fit a degree-1 polynomial (linear regression) of speedup_ratio on
    acceptance_rate. Solve for acceptance_rate where speedup_ratio == 1.0.

    Falls back to the acceptance_rate of the lowest-speedup row with
    speedup_ratio < 1.0 if the data doesn't have enough variation to fit
    a meaningful line (all acceptance rates within 0.01 of each other).
    """
    unique_ar = np.unique(ar_vals)

    if len(unique_ar) < 2 or (unique_ar.max() - unique_ar.min()) < 0.01:
        # Not enough spread — linear regression would be degenerate.
        logger.warning(
            "Acceptance rate values cluster too tightly (range=%.4f) for "
            "meaningful linear regression. Falling back to lowest-speedup "
            "row with speedup_ratio < 1.0.",
            unique_ar.max() - unique_ar.min() if len(unique_ar) > 1 else 0.0,
        )
        below_one = [(ar, sr) for ar, sr in zip(ar_vals, sr_vals) if sr < 1.0]
        if below_one:
            # The acceptance rate at which we're slowest — the conservative fallback.
            fallback_ar = min(below_one, key=lambda x: x[1])[0]
            return round(float(np.clip(fallback_ar, 0.0, 1.0)), 2)
        else:
            # Everything is faster than baseline — break-even is at or below the
            # observed minimum acceptance rate, so report 0.0.
            logger.warning("All speculative rows have speedup_ratio >= 1.0; reporting break_even=0.0")
            return 0.0

    # Fit: speedup_ratio = m * acceptance_rate + b
    # Solve 1.0 = m * ar + b  =>  ar = (1.0 - b) / m
    coeffs = np.polyfit(ar_vals, sr_vals, deg=1)
    m, b = coeffs[0], coeffs[1]

    if abs(m) < 1e-9:
        # Line is essentially flat — can't solve for a crossing.
        logger.warning(
            "Linear fit is nearly flat (slope=%.6f). Falling back to 0.5 as "
            "a neutral break-even estimate.",
            m,
        )
        return 0.5

    break_even_raw = (1.0 - b) / m
    return round(float(np.clip(break_even_raw, 0.0, 1.0)), 2)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute speculative decoding crossover analysis")
    parser.add_argument(
        "--csv",
        default=os.path.join(os.environ.get("RESULTS_DIR", "results"), "sweep_metrics.csv"),
        help="Path to sweep_metrics.csv",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.environ.get("RESULTS_DIR", "results"), "crossover_report.json"),
        help="Output path for crossover_report.json",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found. Run scripts/run_experiments.py first.", file=sys.stderr)
        sys.exit(1)

    result = compute_crossover(args.csv, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
