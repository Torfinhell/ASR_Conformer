from typing import Callable


def lr_by_step(warmup_steps: int, model_dim: int) -> Callable[[int], float]:
    peak_lr = 0.05 / (model_dim**0.5)
    scale = peak_lr * (warmup_steps**0.5)
    return (
        lambda step: min(
            (step if step > 0 else 1) ** (-0.5),
            (step if step > 0 else 1) * warmup_steps ** (-1.5),
        )
        * scale
    )
