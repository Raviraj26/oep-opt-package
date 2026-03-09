import numpy as np
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .workflow import objective
import logging
#logger = logging.getLogger("oep-opt")
grad_logger = logging.getLogger("oep-opt.grad")


#def theta_key(theta: np.ndarray, ndp: int = 12) -> Tuple[float, ...]:
    # robust cache key
#    return tuple(np.round(np.asarray(theta, float), ndp))

def theta_key(theta: np.ndarray) -> bytes:
    # exact float64 bytes => no rounding collisions
    return np.asarray(theta, dtype=np.float64).tobytes()


class Evaluator:
    def __init__(self, cfg, max_workers: int):
        self.cfg = cfg
        self.max_workers = max_workers
        self._cache: Dict[Tuple[float, ...], float] = {}

    def eval_one(self, theta: np.ndarray, phase="grad") -> float:
        k = theta_key(theta)
        if k in self._cache:
            return self._cache[k]
        #self.cfg.logging = False
        val = objective(np.asarray(theta, float), self.cfg,phase=phase)
        #self.cfg.logging = True
        self._cache[k] = float(val)
        return float(val)

    def eval_many(self, thetas: List[np.ndarray]) -> List[float]:
        # ThreadPool is fine because the work is external (Slurm/Molpro); Python GIL not a limiter here.
        out = [None] * len(thetas)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self.eval_one, th): i for i, th in enumerate(thetas)}
            for fut in as_completed(futs):
                i = futs[fut]
                out[i] = fut.result()
        return out
def jac_central_parallel(theta: np.ndarray, evaluator: Evaluator, eps: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    K = theta.size

    grad_logger.info(
        "Computing CD gradient at theta=%s",
        ", ".join(f"{t:.17f}" for t in theta)
    )
    
    thetas = []
    deltas = np.empty(K, dtype=np.float64)

    for i in range(K):
        tp = theta.copy()
        tm = theta.copy()

        #step = theta[i] * eps
        step = eps
        tp[i] += step
        tm[i] -= step

        deltas[i] = 2.0 * step
        thetas.append(tp)
        thetas.append(tm)

    vals = np.asarray(evaluator.eval_many(thetas), dtype=np.float64).reshape(K, 2)
    f_plus = vals[:, 0]
    f_minus = vals[:, 1]

    g = (f_plus - f_minus) / deltas

    grad_logger.info("Gradient: %s", ", ".join(f"{gi:.17f}" for gi in g))
    return g

import numpy as np

def jac_forward_parallel(theta: np.ndarray, evaluator: Evaluator, eps: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    K = theta.size

    grad_logger.info(
        "Computing FD gradient at theta=%s",
        ", ".join(f"{t:.17f}" for t in theta)
    )

    f0 = float(evaluator.eval_one(theta))

    thetas = []
    deltas = np.empty(K, dtype=np.float64)

    for i in range(K):
        tp = theta.copy()

        #step = theta[i] * eps
        step = eps
        tp[i] += step

        deltas[i] = step
        thetas.append(tp)

    f_plus = np.asarray(evaluator.eval_many(thetas), dtype=np.float64)
    g = (f_plus - f0) / deltas

    grad_logger.info("Gradient: %s", ", ".join(f"{gi:.17f}" for gi in g))
    return g

#def jac_central_parallel(theta: np.ndarray, evaluator: Evaluator, eps: float) -> np.ndarray:
#    grad_logger.info(
#        "Computing CD gradient at theta=%s",
#        ", ".join(f"{t:.17f}" for t in theta)
#    )
#    theta = np.asarray(theta, dtype=np.float64)
#    K = theta.size#

#    thetas = np.empty(K, dtype=np.float64)
#    deltas = np.empty(K, dtype=np.float64)
#    for i in range(K):
#        tp = theta.copy()
#        tm = theta.copy()#

#        step = theta[i] * eps
#        tp[i] += step
#        tm[i] -= step

#        deltas[i] = 2.0 * step
#        thetas.append(tp)
#        thetas.append(tm)

    #for i in range(K):
        
    #    tp = theta.copy(); deltas.append(np.float(2.0)*tp[i]*eps); tp[i] += tp[i]*eps
    #    tm = theta.copy(); tm[i] -= tm[i]*eps
    #    thetas.append(tp)
    #    thetas.append(tm)
        

#    vals = evaluator.eval_many(thetas)
#    vals = np.array(vals, float).reshape(K, 2)
#    f_plus = vals[:, 0]
#    f_minus = vals[:, 1]
#    g = []
#    for i in range(K):
#        g[i] = (f_plus[i] - f_minus[i]) / deltas[i] 
#    grad_logger.info("Gradient: %s", ", ".join(f"{gi:.6g}" for gi in g))
#    return g


