"""
test_equivalence.py — Verify that speculative_generate produces the same text
as baseline_generate (target model in greedy mode) for a fixed set of prompts.

If either test fails, the bug is in find_accepted_prefix or the append logic
in speculative_generate. Do not adjust the assertions — fix the source.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logging_config import setup_logging

setup_logging()

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generators import baseline_generate, speculative_generate

# Two fixed prompts — one per domain. Hardcoded here so this test suite has
# no dependency on data/test_prompts.json existing at test time.
ALPACA_PROMPT = "Translate to French: The meeting is scheduled for 3pm."
WRITING_PROMPT = "Write a surprising twist ending for a short story about a lighthouse keeper."

MAX_NEW_TOKENS = 15
N_DRAFT = 4

DRAFT_MODEL_ID = "gpt2"
TARGET_MODEL_ID = "gpt2-large"


@pytest.fixture(scope="module")
def models_and_tokenizer():
    """
    Load models once per test module — gpt2-large is slow to load, and
    reloading it for every test function would make the suite painfully slow.
    """
    cache_dir = os.environ.get("TRANSFORMERS_CACHE", ".hf_cache")

    tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_ID, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_ID, cache_dir=cache_dir)
    draft_model.eval()

    target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, cache_dir=cache_dir)
    target_model.eval()

    return draft_model, target_model, tokenizer


def _check_equivalence(draft_model, target_model, tokenizer, prompt: str) -> None:
    """
    Run both generators on the same prompt and assert text equality.
    The test message deliberately names the mismatch scenario so it's
    obvious which function to look at when debugging.
    """
    result_baseline = baseline_generate(
        target_model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS
    )
    result_speculative = speculative_generate(
        draft_model, target_model, tokenizer, prompt,
        n_draft=N_DRAFT, max_new_tokens=MAX_NEW_TOKENS,
    )

    baseline_text = result_baseline["text"]
    speculative_text = result_speculative["text"]

    assert baseline_text == speculative_text, (
        f"Output mismatch for prompt: {prompt!r}\n"
        f"  baseline:    {baseline_text!r}\n"
        f"  speculative: {speculative_text!r}\n"
        "Check find_accepted_prefix index math and the token-append logic "
        "in speculative_generate."
    )


def test_equivalence_alpaca_prompt(models_and_tokenizer):
    """Instruction-style prompt (alpaca domain) — high acceptance rate expected."""
    draft_model, target_model, tokenizer = models_and_tokenizer
    _check_equivalence(draft_model, target_model, tokenizer, ALPACA_PROMPT)


def test_equivalence_writing_prompt(models_and_tokenizer):
    """Open-ended creative prompt (writing_prompts domain) — lower acceptance rate expected."""
    draft_model, target_model, tokenizer = models_and_tokenizer
    _check_equivalence(draft_model, target_model, tokenizer, WRITING_PROMPT)
