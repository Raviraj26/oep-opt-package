# Placeholder for scoring.py. Full code was in assistant's answer.
import numpy as np
from typing import Dict, Optional, Sequence
from .config import Weights, S_ovrlp_penalty, Redundancy_penalty, A_coupling_penalty
from .config import JobConfig
import logging
import json
#logger = logging.getLogger("oep-opt")

def eval_s_ovrlp_penalty(phase, logger, sc, first_eig_of_S, expo, coeff):
    log_first_eig_of_S = -1.0*np.log10(first_eig_of_S)
    if log_first_eig_of_S <= expo:
        return sc
    elif log_first_eig_of_S > expo:
        s_ovrlp_penalty = coeff * ( log_first_eig_of_S - expo ) ** 4
        if phase == "log":
            logger.info("Penalty for small e_val of S %s", s_ovrlp_penalty)
        return sc + s_ovrlp_penalty

def eval_a_coupling_penalty(phase, logger, sc, first_eig_of_A, expo, coeff):
    log_first_eig_of_A = -1.0*np.log10(first_eig_of_A)
    #if logging:
    if phase == "log":
        logger.info("Eigen value considered in panalty %s", first_eig_of_A)
    if log_first_eig_of_A <= expo:
        return sc
    elif log_first_eig_of_A > expo:
        a_coupling_penalty = coeff * ( log_first_eig_of_A - expo ) ** 4
        #if logging:
        if phase == "log":
            logger.info("Penalty for small e_val of A %s", a_coupling_penalty)
        return sc + a_coupling_penalty


def eval_logdet_penalty(phase, logger, sc, all_eig_of_A, coeff, P0):
    """
    Log-determinant barrier: penalty = coeff * (-ln det A) / P0
    
    all_eig_of_A : list/array of all A^{III} eigenvalues
    coeff        : penalty strength (equals penalty value at seed)
    P0           : -ln(det A) at seed, for normalization
    """
    eigs = np.array(all_eig_of_A, dtype=np.float64)
    eigs_safe = np.maximum(eigs, 1e-30)
    neg_log_det = -np.sum(np.log(eigs_safe))
    
    if P0 is not None and abs(P0) > 1e-30:
        penalty = coeff * neg_log_det / P0
    else:
        penalty = coeff * neg_log_det
    
    if phase == "log":
        logger.info("logdet penalty: -ln(det A) = %.6f, P0 = %.6f, "
                     "normalized penalty = %.6f, lambda_min = %.4e",
                     neg_log_det, P0 if P0 else 0.0, penalty, eigs[0])
    return sc + penalty
    
def eval_redundancy_penalty(phase,logger, sc, exps_desc, a, b, c):
    exps_list = list(exps_desc)
    ratio_exps_list = []
    redundancy_penalty_value = 0.0
    for i in range(0, len(exps_list)-1):
        ratio_val = exps_list[i] / exps_list[i+1]
        #logger.info("i is %s exps[i] is %s exps[i+1] is %s ratio is %s", i, exps_list[i], exps_list[i+1], ratio_val)
        ratio_exps_list.append(ratio_val)
    #if logging:
    if phase == "log":
        logger.info("ratio list: %s", ",".join(f"{t:.4f}" for t in ratio_exps_list))
    for i in range(0, len(ratio_exps_list)):
        if ratio_exps_list[i] < a:
            redundancy_penalty_value = redundancy_penalty_value + c * (b*(a-ratio_exps_list[i]))**2
        #if logging:
        if phase == "log":
            logger.info("i is %s, ratio[i] is %s, redundancy_penalty_value is %s", i, ratio_exps_list[i], redundancy_penalty_value)
    return sc + redundancy_penalty_value
        #ratio = 0.0
        #ratio = ratio + exps_list[]


def negative_exps_penalty(exps):
    penalty = 0.0
    for i in range(0, len(exps)):
        if exps[i] < 0.0:
            penalty = penalty + 1.0 * (exps[i])**(-4)
    return penalty

def get_normalised_metrics(seed_metrics, metrics):
    normalised_metrics = {}
    list_to_normalise = ["dnorm", "rscaled_dnorm", "sqrtrscaled_dnorm","rtimes_scaled_dnorm","rsqr_scaled_dnorm",
                         "ref_proj_dnorm", "ref_proj_rscaled_dnorm", "ref_proj_sqrtrscaled_dnorm", "ref_proj_rtimes_scaled_dnorm", "ref_proj_rsqr_scaled_dnorm"]
    for key in metrics.keys():
        seed_val = seed_metrics.get(key)
        current_val = metrics.get(key)
        if key in list_to_normalise:
            if seed_val is not None and current_val is not None and seed_val != 0.0:
                normalised_metrics[key] = current_val / seed_val
            else:
                normalised_metrics[key] = None
        else:
            normalised_metrics[key] = current_val
    return normalised_metrics

_NORM_KEYS = [
    "dnorm", "rscaled_dnorm", "sqrtrscaled_dnorm",
    "rtimes_scaled_dnorm", "rsqr_scaled_dnorm",
    "ref_proj_dnorm", "ref_proj_rscaled_dnorm",
    "ref_proj_sqrtrscaled_dnorm", "ref_proj_rtimes_scaled_dnorm",
    "ref_proj_rsqr_scaled_dnorm",
]

def load_or_save_seed_metrics(cfg: JobConfig, metrics: dict,
                                logger_file) -> dict:
    """
    On first successful call: write seed_metrics.json and return metrics.
    On subsequent calls: load from disk and return.
    Persists across process restarts — safe for SLURM job arrays.
    """
    seed_path = cfg.workroot / "seed_metrics.json"
    if seed_path.exists():
        with open(seed_path) as fh:
            seed = json.load(fh)
        logger_file.info("Seed metrics loaded from %s", seed_path)
        return seed
    # First successful run — write to disk
    seed = {k: metrics[k] for k in _NORM_KEYS if metrics.get(k) is not None}
    # Store logdet P0 for normalization of log-determinant penalty
    all_eig = metrics.get("all_eig_of_A")
    if all_eig is not None:
        eigs = np.array(all_eig, dtype=np.float64)
        seed["logdet_P0"] = float(-np.sum(np.log(np.maximum(eigs, 1e-30))))
    with open(seed_path, "w") as fh:
        json.dump(seed, fh, indent=2)
    logger_file.info("Seed metrics saved to %s: %s", seed_path,
                     ", ".join(f"{k}={v:.6g}" for k, v in seed.items()))
    return seed

def score_from_metrics(exps: Sequence[float], metrics: Dict[str, Optional[float]], weights: Weights,
                        s_ovrlp_penalty: S_ovrlp_penalty, redundancy_penalty: Redundancy_penalty, a_coupling_penalty: A_coupling_penalty,
                        fail_penalty: float = 1e6, phase ="log", seed_metrics: Optional[Dict] = None) -> float:
    dv, du, dlieb = (metrics.get(k) for k in ("dvext", "du", "dlieb"))
    dnorm, rscaled_dnorm, sqrtrscaled_dnorm, rtimes_scaled_dnorm, rsqr_scaled_dnorm = (metrics.get(k) for k in ("dnorm", "rscaled_dnorm", "sqrtrscaled_dnorm", "rtimes_scaled_dnorm","rsqr_scaled_dnorm"))
    ref_proj_dnorm, ref_proj_rscaled_dnorm, ref_proj_sqrtrscaled_dnorm, ref_proj_rtimes_scaled_dnorm, ref_proj_rsqr_scaled_dnorm = (metrics.get(k) for k in ("ref_proj_dnorm", "ref_proj_rscaled_dnorm", "ref_proj_sqrtrscaled_dnorm", "ref_proj_rtimes_scaled_dnorm", "ref_proj_rsqr_scaled_dnorm"))
    logger = logging.getLogger("oep-opt") if phase == "log" else logging.getLogger("oep-opt.grad")
    terms = []
    if dv is not None:    terms.append(weights.w_dvext * abs(dv))
    if du is not None:    terms.append(weights.w_du    * abs(du))
    if dlieb is not None: terms.append(weights.w_lieb  * abs(dlieb))
    if dnorm is not None: terms.append(weights.w_norm  * abs(dnorm))
    if rscaled_dnorm is not None: terms.append(weights.w_rscaled_norm  * abs(rscaled_dnorm))
    if sqrtrscaled_dnorm is not None: terms.append(weights.w_sqrtrscaled_norm  * abs(sqrtrscaled_dnorm))
    if rtimes_scaled_dnorm is not None: terms.append(weights.w_rtimes_scaled_norm  * abs(rtimes_scaled_dnorm))
    if rsqr_scaled_dnorm is not None: terms.append(weights.w_rsqr_scaled_norm  * abs(rsqr_scaled_dnorm))
    if ref_proj_dnorm is not None: terms.append(weights.w_ref_proj_norm  * abs(ref_proj_dnorm))
    if ref_proj_rscaled_dnorm is not None: terms.append(weights.w_ref_proj_rscaled_norm  * abs(ref_proj_rscaled_dnorm))
    if ref_proj_sqrtrscaled_dnorm is not None: terms.append(weights.w_ref_proj_sqrtrscaled_norm  * abs(ref_proj_sqrtrscaled_dnorm))
    if ref_proj_rtimes_scaled_dnorm is not None: terms.append(weights.w_ref_proj_rtimes_scaled_norm  * abs(ref_proj_rtimes_scaled_dnorm))
    if ref_proj_rsqr_scaled_dnorm is not None: terms.append(weights.w_ref_proj_rsqr_scaled_norm  *  abs(ref_proj_rsqr_scaled_dnorm))

    first_eig_of_S = metrics.get("first_eig_of_S")
    #first_eig_of_A = metrics.get("first_eig_of_A")
    opt_first_eig_of_A = metrics.get("opt_first_eig_of_A")
    if not terms: return fail_penalty
    sc = float(sum(terms))
    if phase == "log":
        logger.info("Score components: %s", ", ".join(f"{t:.12f}" for t in terms))
        logger.info("Total score: %.12f", sc)
    
    if s_ovrlp_penalty.knob:
        sc = eval_s_ovrlp_penalty(phase, logger, sc, first_eig_of_S, s_ovrlp_penalty.expo, s_ovrlp_penalty.coeff)  
        #sc_w_s_ovrlp_penalty = eval_s_ovrlp_penalty(sc, first_eig_of_S, s_ovrlp_penalty.expo, s_ovrlp_penalty.coeff)
        if phase == "log":
            logger.info("Total score with s_overlp penalty: %.12f", sc)
    
    if redundancy_penalty.knob:
        sc = eval_redundancy_penalty(phase, logger, sc, exps, redundancy_penalty.a,redundancy_penalty.b,redundancy_penalty.c)
        if phase == "log":
            logger.info("Total score with redundancy penalty: %.12f", sc)
    
    if a_coupling_penalty.knob:
        if a_coupling_penalty.penalty_type == "logdet":
            all_eig = metrics.get("all_eig_of_A")
            if all_eig is not None:
                P0 = seed_metrics.get("logdet_P0") if seed_metrics else None
                sc = eval_logdet_penalty(phase, logger, sc, all_eig,
                                         a_coupling_penalty.coeff, P0)
            else:
                if phase == "log":
                    logger.warning("all_eig_of_A is None — cannot compute logdet penalty. "
                                   "Ensure VERB,3 is set in KSINV input.")
        else:
            sc = eval_a_coupling_penalty(phase, logger, sc, opt_first_eig_of_A,
                                          a_coupling_penalty.expo, a_coupling_penalty.coeff)
        if phase == "log":
            logger.info("Total score with A coupling penalty: %.12f", sc)

    conv = metrics.get("converged")
    #if conv is False: sc += weights.penalty_noconv
    #elif conv is None: sc += 0.1 * weights.penalty_noconv
    if not np.isfinite(sc): return fail_penalty


    return sc