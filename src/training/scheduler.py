from __future__ import annotations


def linear_temperature(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return end
    ratio = min(max(step / total_steps, 0.0), 1.0)
    return start + (end - start) * ratio

