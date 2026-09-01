"""
generators.py — Baseline and speculative greedy text generators.

Neither function uses the HuggingFace high-level generation API. Both drive
the forward pass manually so we have full control over what gets measured and
logged.
"""

import logging
import time
from typing import Dict, Union

import torch

from src.verify import find_accepted_prefix

logger = logging.getLogger(__name__)


def baseline_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> Dict[str, Union[str, float]]:
    """
    Greedy autoregressive generation using a manual token-by-token loop with
    KV caching. No sampling, no beam search — pure argmax at each step.

    Returns a dict with keys:
      text          — generated text, excluding the prompt itself
      latency       — wall-clock seconds for the whole generation
      tokens_per_sec — new tokens generated per second
    """
    model.eval()
    device = next(model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_len = input_ids.shape[1]
    eos_id = tokenizer.eos_token_id

    generated_ids = []
    past_key_values = None

    t0 = time.perf_counter()

    with torch.no_grad():
        # First pass covers the full prompt; subsequent passes feed one token.
        current_ids = input_ids
        current_mask = attention_mask

        for _ in range(max_new_tokens):
            out = model(
                input_ids=current_ids,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            next_token_id = int(out.logits[:, -1, :].argmax(-1))
            generated_ids.append(next_token_id)

            if next_token_id == eos_id:
                break

            # From the second iteration onward we only feed the new token.
            current_ids = torch.tensor([[next_token_id]], device=device)
            if current_mask is not None:
                # Extend the mask by one position.
                current_mask = torch.cat(
                    [current_mask, torch.ones((1, 1), device=device, dtype=current_mask.dtype)],
                    dim=1,
                )

    latency = time.perf_counter() - t0
    n_new = len(generated_ids)
    tokens_per_sec = n_new / latency if latency > 0 else 0.0

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    logger.debug(
        "baseline_generate: %d new tokens in %.3fs (%.1f tok/s)",
        n_new, latency, tokens_per_sec,
    )

    return {
        "text": text,
        "latency": latency,
        "tokens_per_sec": tokens_per_sec,
    }


def speculative_generate(
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    n_draft: int,
    max_new_tokens: int,
) -> Dict[str, Union[str, float]]:
    """
    Speculative decoding: the draft model proposes n_draft tokens per round;
    the target model verifies them in one forward pass; accepted tokens plus
    a corrective/bonus token are appended; repeat.

    Correctness note: the target model always runs a full sequence recompute
    each round (no KV cache across the draft/verify boundary). This is slower
    than a cached version but makes the equivalence invariant easy to reason
    about — baseline and speculative agree because neither skips a target logit.

    Returns a dict with keys:
      text            — generated text, excluding the prompt
      latency         — wall-clock seconds
      tokens_per_sec  — new tokens per second
      acceptance_rate — accepted_draft_tokens / total_proposed_draft_tokens
                        (0.0 if no draft round completed, not NaN)
    """
    draft_model.eval()
    target_model.eval()
    device = next(target_model.parameters()).device
    draft_device = next(draft_model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt")
    prompt_ids = enc["input_ids"]  # kept on CPU initially, moved per-model
    eos_id = tokenizer.eos_token_id

    # Running sequence of all token ids (prompt + generated).
    sequence_ids: list = prompt_ids[0].tolist()
    prompt_len = len(sequence_ids)

    total_proposed = 0
    total_accepted = 0
    generated_count = 0

    t0 = time.perf_counter()

    with torch.no_grad():
        while generated_count < max_new_tokens:
            tokens_remaining = max_new_tokens - generated_count
            # Don't propose more tokens than we're allowed to generate.
            actual_n_draft = min(n_draft, tokens_remaining)
            if actual_n_draft <= 0:
                break

            base_len = len(sequence_ids)

            # --- Draft phase: autoregressively sample actual_n_draft tokens ---
            draft_input = torch.tensor([sequence_ids], dtype=torch.long, device=draft_device)
            draft_past = None
            draft_ids = []

            # First draft step: feed full sequence to warm the KV cache.
            draft_current = draft_input
            for d_step in range(actual_n_draft):
                d_out = draft_model(
                    input_ids=draft_current,
                    past_key_values=draft_past,
                    use_cache=True,
                )
                draft_past = d_out.past_key_values
                d_token = int(d_out.logits[:, -1, :].argmax(-1))
                draft_ids.append(d_token)

                if d_token == eos_id:
                    break

                draft_current = torch.tensor([[d_token]], dtype=torch.long, device=draft_device)

            n_proposed = len(draft_ids)
            total_proposed += n_proposed

            # --- Verify phase: one target forward pass over prefix + draft ---
            verify_input = torch.tensor(
                [sequence_ids + draft_ids], dtype=torch.long, device=device
            )
            t_out = target_model(input_ids=verify_input, use_cache=False)
            # t_out.logits shape: (1, base_len + n_proposed, vocab_size)

            accepted, corrective_or_bonus = find_accepted_prefix(
                t_out.logits, draft_ids, base_len
            )

            logger.debug(
                "speculative step: proposed=%d accepted=%d corrective=%s",
                n_proposed, accepted, corrective_or_bonus,
            )

            total_accepted += accepted

            # Append accepted draft tokens plus the corrective/bonus token.
            new_tokens = draft_ids[:accepted]
            if corrective_or_bonus is not None:
                new_tokens.append(corrective_or_bonus)

            # Clip to the remaining budget. When tokens_remaining=1 and the
            # draft token was accepted, we'd have [accepted_tok, bonus_tok]=2
            # tokens but only 1 slot left — the bonus gets dropped here.
            # Baseline stops at exactly max_new_tokens, so we must too.
            new_tokens = new_tokens[:tokens_remaining]

            sequence_ids.extend(new_tokens)
            generated_count += len(new_tokens)

            # Check for EOS anywhere in the newly appended tokens.
            eos_hit = False
            for tok in new_tokens:
                if tok == eos_id:
                    eos_hit = True
                    break
            if eos_hit:
                break

    latency = time.perf_counter() - t0
    n_new = len(sequence_ids) - prompt_len
    tokens_per_sec = n_new / latency if latency > 0 else 0.0
    acceptance_rate = (
        total_accepted / total_proposed if total_proposed > 0 else 0.0
    )

    generated_ids = sequence_ids[prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    logger.debug(
        "speculative_generate done: %d new tokens in %.3fs (%.1f tok/s) "
        "acceptance_rate=%.3f",
        n_new, latency, tokens_per_sec, acceptance_rate,
    )

    return {
        "text": text,
        "latency": latency,
        "tokens_per_sec": tokens_per_sec,
        "acceptance_rate": acceptance_rate,
    }
