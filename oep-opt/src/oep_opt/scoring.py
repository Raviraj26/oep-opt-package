# Placeholder for scoring.py. Full code was in assistant's answer.
import numpy as np
from typing import Dict, Optional
from .config import Weights
import logging
logger = logging.getLogger("oep-opt")



def score_from_metrics(metrics: Dict[str, Optional[float]], weights: Weights, fail_penalty: float = 1e6) -> float:
    dv, du, dlieb, dnorm = (metrics.get(k) for k in ("dvext", "du", "dlieb", "dnorm"))
    terms = []
    if dv is not None:    terms.append(weights.w_dvext * abs(dv))
    if du is not None:    terms.append(weights.w_du    * abs(du))
    if dlieb is not None: terms.append(weights.w_lieb  * abs(dlieb))
    if dnorm is not None: terms.append(weights.w_norm  * abs(dnorm))
    if not terms: return fail_penalty
    sc = float(sum(terms))
    logger.info("Score components: %s", ", ".join(f"{t:.12f}" for t in terms))
    logger.info("Total score: %.12f", sc)
    conv = metrics.get("converged")
    if conv is False: sc += weights.penalty_noconv
    elif conv is None: sc += 0.1 * weights.penalty_noconv
    if not np.isfinite(sc): return fail_penalty
    return sc
