# Placeholder for parameterizations.py. Full code was in assistant's answer.
from typing import List, Sequence
import numpy as np

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def ensure_descending(exps: Sequence[float]) -> List[float]:
    out = list(sorted(exps, reverse=True))
    for i in range(1, len(out)):
        if out[i] >= out[i-1]:
            out[i] = max(1e-12, out[i-1] * 0.9999)
    return out

def even_tempered_from_params(theta: np.ndarray, K: int, exp_min: float, exp_max: float) -> List[float]:
    u, v = float(theta[0]), float(theta[1])
    alpha_hi = float(_softplus(np.array([u]))[0]) + 1e-8
    beta = 1.0 + float(_softplus(np.array([v]))[0])
    exps = [alpha_hi * (beta ** (-k)) for k in range(K)]
    exps = [min(max(x, exp_min), exp_max) for x in exps]
    return exps

def free_exponents_from_params(theta: np.ndarray, K: int, exp_min: float, exp_max: float) -> List[float]:
    if len(theta) != K:
        raise ValueError(f"theta length {len(theta)} must equal K={K}.")
    exps = list(np.exp(theta))
    exps = [min(max(x, exp_min), exp_max) for x in exps]
    return exps

