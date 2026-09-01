"""
verify.py — Logit comparison logic for speculative decoding.

The key insight: the target model's forward pass over [prefix + N draft tokens]
produces N+1 useful logit positions at no extra cost. Positions [base_len-1 ..
base_len-1+N-1] predict the N drafted tokens; position [base_len-1+N] gives us
a free (N+1)-th token if all N were accepted. We walk left to right and bail out
at the first mismatch.
"""

import torch
from typing import Optional, Tuple, Union


def find_accepted_prefix(
    target_logits: torch.Tensor,
    draft_token_ids: Union[list, torch.Tensor],
    base_len: int,
) -> Tuple[int, Optional[int]]:
    """
    Walk the draft tokens left to right, comparing each one to the target
    model's argmax at the corresponding logit position.

    target_logits: tensor of shape (1, base_len + N, vocab_size) — full target
                   forward pass over [prefix (len=base_len)] + [N draft tokens].
    draft_token_ids: list or tensor of length N — the drafted token ids.
    base_len: int — number of tokens in the sequence *before* the draft window.

    Position arithmetic (0-indexed):
      - Logit at index (base_len - 1 + k) predicts the token at sequence
        position (base_len + k), i.e., draft token k.
      - So for k in [0, N): compare target_logits[:, base_len-1+k, :].argmax(-1)
        to draft_token_ids[k].

    Returns (accepted_count, corrective_token_id):
      accepted_count     — how many leading draft tokens matched target argmax.
      corrective_token_id — target's own token at the first mismatch position
                            (the one we swap in). None if all N were accepted,
                            in which case we instead return the bonus token from
                            the position *after* the last draft slot (base_len-1+N).
                            That bonus token is the (N+1)-th token you get for
                            free from the same forward pass.
    """
    if isinstance(draft_token_ids, torch.Tensor):
        draft_ids = draft_token_ids.tolist()
    else:
        draft_ids = list(draft_token_ids)

    n = len(draft_ids)

    for k in range(n):
        logit_pos = base_len - 1 + k
        target_token = int(target_logits[:, logit_pos, :].argmax(-1))
        if target_token != draft_ids[k]:
            # Mismatch at position k — use the target's choice here and discard
            # all remaining draft tokens.
            return k, target_token

    # All N draft tokens were accepted. The logit at base_len-1+N is the bonus
    # token — it's what the target model would have predicted next anyway.
    bonus_pos = base_len - 1 + n
    bonus_token = int(target_logits[:, bonus_pos, :].argmax(-1))
    return n, bonus_token
