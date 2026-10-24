"""Schedules: probability of the new tokenizer and learning rate."""
from __future__ import annotations

import math


def p_new(step: int, total_steps: int, rho: float) -> float:
    """p(t) = min(1, t / (rho T)). rho = 0 is the abrupt swap (p = 1 from step 0)."""
    if rho <= 0:
        return 1.0
    return min(1.0, step / (rho * total_steps))


def lr_at(step: int, total_steps: int, peak: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay to ``min_ratio * peak``."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
