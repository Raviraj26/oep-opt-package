# Placeholder for scoring.py. Full code was in assistant's answer.
import numpy as np
from typing import Dict, Optional, Sequence
from .config import Weights, S_ovrlp_penalty, Redundancy_penalty, A_coupling_penalty
import logging
logger = logging.getLogger("oep-opt")

def eval_s_ovrlp_penalty(sc, first_eig_of_S, expo, coeff):
    log_first_eig_of_S = -1.0*np.log10(first_eig_of_S)
    if log_first_eig_of_S <= expo:
        return sc
    elif log_first_eig_of_S > expo:
        s_ovrlp_penalty = coeff * ( log_first_eig_of_S - expo ) ** 2
        logger.info("Penalty for small e_val of S %s", s_ovrlp_penalty)
        return sc + s_ovrlp_penalty

def eval_a_coupling_penalty(sc, first_eig_of_A, expo, coeff):
    log_first_eig_of_A = -1.0*np.log10(first_eig_of_A)
    logger.info("Eigen value considered in panalty %s", first_eig_of_A)
    if log_first_eig_of_A <= expo:
        return sc
    elif log_first_eig_of_A > expo:
        a_coupling_penalty = coeff * ( log_first_eig_of_A - expo ) ** 2
        logger.info("Penalty for small e_val of A %s", a_coupling_penalty)
        return sc + a_coupling_penalty
    
def eval_redundancy_penalty(sc, exps_desc, a, b, c):
    exps_list = list(exps_desc)
    ratio_exps_list = []
    redundancy_penalty_value = 0.0
    for i in range(0, len(exps_list)-1):
        ratio_val = exps_list[i] / exps_list[i+1]
        #logger.info("i is %s exps[i] is %s exps[i+1] is %s ratio is %s", i, exps_list[i], exps_list[i+1], ratio_val)
        ratio_exps_list.append(ratio_val)
    
    logger.info("ratio list: %s", ",".join(f"{t:.4f}" for t in ratio_exps_list))
    for i in range(0, len(ratio_exps_list)):
        if ratio_exps_list[i] < a:
            redundancy_penalty_value = redundancy_penalty_value + c * (b*(a-ratio_exps_list[i]))**2
        logger.info("i is %s, ratio[i] is %s, redundancy_penalty_value is %s", i, ratio_exps_list[i], redundancy_penalty_value)
    return sc + redundancy_penalty_value
        #ratio = 0.0
        #ratio = ratio + exps_list[]


def score_from_metrics(exps_desc: Sequence[float], metrics: Dict[str, Optional[float]], weights: Weights,
                        s_ovrlp_penalty: S_ovrlp_penalty, redundancy_penalty: Redundancy_penalty, a_coupling_penalty: A_coupling_penalty,
                        fail_penalty: float = 1e6) -> float:
    dv, du, dlieb, dnorm, rscaled_dnorm, sqrtrscaled_dnorm, rtimes_scaled_dnorm = (metrics.get(k) for k in ("dvext", "du", "dlieb", "dnorm", "rscaled_dnorm", "sqrtrscaled_dnorm", "rtimes_scaled_dnorm",))
    terms = []
    if dv is not None:    terms.append(weights.w_dvext * abs(dv))
    if du is not None:    terms.append(weights.w_du    * abs(du))
    if dlieb is not None: terms.append(weights.w_lieb  * abs(dlieb))
    if dnorm is not None: terms.append(weights.w_norm  * abs(dnorm))
    if rscaled_dnorm is not None: terms.append(weights.w_rscaled_norm  * abs(rscaled_dnorm))
    if sqrtrscaled_dnorm is not None: terms.append(weights.w_sqrtrscaled_norm  * abs(sqrtrscaled_dnorm))
    if rtimes_scaled_dnorm is not None: terms.append(weights.w_rtimes_scaled_norm  * abs(rtimes_scaled_dnorm))
    first_eig_of_S = metrics.get("first_eig_of_S")
    #first_eig_of_A = metrics.get("first_eig_of_A")
    opt_first_eig_of_A = metrics.get("opt_first_eig_of_A")
    if not terms: return fail_penalty
    sc = float(sum(terms))
    logger.info("Score components: %s", ", ".join(f"{t:.12f}" for t in terms))
    logger.info("Total score: %.12f", sc)
    
    if s_ovrlp_penalty.knob:
        sc = eval_s_ovrlp_penalty(sc, first_eig_of_S, s_ovrlp_penalty.expo, s_ovrlp_penalty.coeff)  
        #sc_w_s_ovrlp_penalty = eval_s_ovrlp_penalty(sc, first_eig_of_S, s_ovrlp_penalty.expo, s_ovrlp_penalty.coeff)
        logger.info("Total score with s_overlp penalty: %.12f", sc)
    
    if redundancy_penalty.knob:
        sc = eval_redundancy_penalty(sc, exps_desc, redundancy_penalty.a,redundancy_penalty.b,redundancy_penalty.c)
        logger.info("Total score with redundancy penalty: %.12f", sc)
    
    if a_coupling_penalty.knob:
        sc = eval_a_coupling_penalty(sc, opt_first_eig_of_A, a_coupling_penalty.expo, a_coupling_penalty.coeff)
        logger.info("Total score with A coupling penalty: %.12f", sc)

    conv = metrics.get("converged")
    #if conv is False: sc += weights.penalty_noconv
    #elif conv is None: sc += 0.1 * weights.penalty_noconv
    if not np.isfinite(sc): return fail_penalty


    return sc
