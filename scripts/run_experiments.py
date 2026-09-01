"""
run_experiments.py — Sweep baseline and speculative generation over test prompts.

Usage:
    python scripts/run_experiments.py [--prompts PATH] [--limit N]

Default: processes the first 20 prompts per domain (40 total). Pass --limit 0
to run the full dataset, but expect several hours on CPU for 200 prompts with
gpt2-large as the target model.
"""

import argparse
import json
import os
import sys
import time

# Make src importable when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generators import baseline_generate, speculative_generate
from src.logging_config import setup_logging

setup_logging()

import logging
logger = logging.getLogger(__name__)

N_DRAFT_VALUES = [2, 4, 8]
MAX_NEW_TOKENS = 30
DEFAULT_LIMIT_PER_DOMAIN = 20


def load_config(config_path: str = "submission.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_prompts(path: str, limit_per_domain: int) -> list:
    with open(path, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)

    if limit_per_domain <= 0:
        return all_prompts

    by_domain: dict = {}
    for p in all_prompts:
        domain = p["domain"]
        by_domain.setdefault(domain, []).append(p)

    selected = []
    for domain, items in by_domain.items():
        selected.extend(items[:limit_per_domain])

    return selected


def run_sweep(prompts: list, draft_model, target_model, tokenizer) -> list:
    rows = []

    for entry in prompts:
        pid = entry["id"]
        domain = entry["domain"]
        prompt = entry["prompt"]

        logger.info("Running baseline for prompt %s (%s)", pid, domain)
        bl = baseline_generate(target_model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
        rows.append({
            "prompt_id": pid,
            "domain": domain,
            "method": "baseline",
            "n_draft": 0,
            "latency_sec": bl["latency"],
            "tokens_per_sec": bl["tokens_per_sec"],
            "acceptance_rate": 1.0,
        })

        for n in N_DRAFT_VALUES:
            logger.info("Running speculative n_draft=%d for prompt %s", n, pid)
            sp = speculative_generate(
                draft_model, target_model, tokenizer, prompt,
                n_draft=n, max_new_tokens=MAX_NEW_TOKENS,
            )
            rows.append({
                "prompt_id": pid,
                "domain": domain,
                "method": "speculative",
                "n_draft": n,
                "latency_sec": sp["latency"],
                "tokens_per_sec": sp["tokens_per_sec"],
                "acceptance_rate": sp["acceptance_rate"],
            })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Speculative decoding experiment sweep")
    parser.add_argument(
        "--prompts",
        default="data/test_prompts.json",
        help="Path to test_prompts.json (default: data/test_prompts.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT_PER_DOMAIN,
        help=(
            f"Max prompts per domain to process. Default {DEFAULT_LIMIT_PER_DOMAIN}. "
            "Pass 0 to run the full dataset (slow on CPU)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=os.environ.get("RESULTS_DIR", "results"),
        help="Directory to write sweep_metrics.csv (default: results/)",
    )
    args = parser.parse_args()

    config = load_config()
    draft_id = config["model_configs"]["draft_model_id"]
    target_id = config["model_configs"]["target_model_id"]

    prompts = load_prompts(args.prompts, args.limit)
    n_total = len(prompts)
    limit_msg = (
        f"all {n_total}"
        if args.limit == 0
        else f"{n_total} (first {args.limit} per domain — pass --limit 0 for full 200)"
    )
    print(f"Running sweep over {limit_msg} prompts with draft={draft_id}, target={target_id}")
    print(f"n_draft values: {N_DRAFT_VALUES}, max_new_tokens={MAX_NEW_TOKENS}")

    cache_dir = os.environ.get("TRANSFORMERS_CACHE", ".hf_cache")

    print(f"Loading tokenizer from {draft_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(draft_id, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading draft model ({draft_id}) ...")
    draft_model = AutoModelForCausalLM.from_pretrained(draft_id, cache_dir=cache_dir)
    draft_model.eval()

    print(f"Loading target model ({target_id}) — this may take a moment on CPU ...")
    target_model = AutoModelForCausalLM.from_pretrained(target_id, cache_dir=cache_dir)
    target_model.eval()

    t_start = time.perf_counter()
    rows = run_sweep(prompts, draft_model, target_model, tokenizer)
    elapsed = time.perf_counter() - t_start

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "sweep_metrics.csv")
    df = pd.DataFrame(rows, columns=[
        "prompt_id", "domain", "method", "n_draft",
        "latency_sec", "tokens_per_sec", "acceptance_rate",
    ])
    df.to_csv(out_path, index=False)

    print(f"\nDone. Total wall time: {elapsed:.1f}s")
    print(f"Results written to: {out_path}")
    print(df.groupby(["domain", "method", "n_draft"])[["tokens_per_sec", "acceptance_rate"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
