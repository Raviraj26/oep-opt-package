# Placeholder for workflow.py. Full code was in assistant's answer.
import json
from pathlib import Path
import numpy as np
from typing import List
import logging
logger = logging.getLogger("oep-opt")

grad_logger = logging.getLogger("oep-opt.grad")


from .config import JobConfig
from .parameterizations import even_tempered_from_params, free_exponents_from_params
from .io_utils import format_exps_for_molpro, write_input_file, stage_dm_as_link
from .slurm import run_molpro_via_slurm
from .parsing import parse_metrics
from .scoring import score_from_metrics
from .scoring import get_normalised_metrics, load_or_save_seed_metrics
from .utils import stable_tag_from_theta
from .scoring import negative_exps_penalty

def exps_from_theta(theta: np.ndarray, cfg: JobConfig) -> List[float]:
    if cfg.mode == "even_tempered":
        return even_tempered_from_params(theta, cfg.K, cfg.exp_min, cfg.exp_max)
    elif cfg.mode == "free_exponents":
        return free_exponents_from_params(theta, cfg.K, cfg.exp_min, cfg.exp_max)
    raise ValueError(f"Unknown mode: {cfg.mode}")

def objective(theta: np.ndarray, cfg: JobConfig, phase = "log") -> float:
    tag = stable_tag_from_theta(theta)
    rundir = cfg.workroot / f"run_{cfg.elem}_{cfg.mode}_{tag}"
    rundir.mkdir(parents=True, exist_ok=True)
    logger_file = logger if phase == "log" else grad_logger
    #dm_src = Path(cfg.dm_file)
    #if dm_src.exists():
    #    tgt = rundir / dm_src.name
    #    if not tgt.exists():
    #        tgt.write_bytes(dm_src.read_bytes())
    stage_dm_as_link(cfg.dm_file, rundir)

    exps = exps_from_theta(theta, cfg)
    #exps = theta


    #if phase == "log":
    #    logger.info("exps dtype=%s (item type=%s)", exps.dtype, type(exps.flat[0]).__name__)

    #exps_desc = exps

    s_line = format_exps_for_molpro(exps)
    inp_path = write_input_file(cfg.template_text, cfg.elem, cfg.charge, cfg.spin,
                                cfg.alpha_occ, cfg.beta_occ, cfg.r_dnormcutoff,
                                cfg.orbital_parent, cfg.aux_parent, s_line,
                                cfg.dm_file, cfg.e_ref, rundir)

#    try:
#        logger.info("TRY %s: mode=%s, K=%d -> %s", cfg.elem, cfg.mode, cfg.K, rundir.name)
#        print(f"[TRY] {cfg.elem}: mode={cfg.mode}, K={cfg.K}, exps[0]={exps_desc[0]:.3f} -> {rundir.name}")
#    except Exception:
#        print(f"[TRY] {cfg.elem}: mode={cfg.mode}, K={cfg.K} -> {rundir.name}")
    
    try:
        logger_file.info(
            " %s: mode=%s, K=%d: %s -> %s",
            cfg.elem,
            cfg.mode,
            cfg.K,
            ", ".join(f"{e:.17f}" for e in exps),
            rundir.name,
        )
    except Exception:
        logger_file.info(
            " %s: mode=%s, K=%d -> %s",
            cfg.elem,
            cfg.mode,
            cfg.K,
            rundir.name,
        )

    #for i in range(0, len(exps)):
    #    if exps[i] < 0.0:
    #        sc = negative_exps_penalty(exps)
    #        if phase == "log":
    #            logger.info("Penalty for negative exps %s", sc)
    #        return sc
    init_run = True
    rc, out_text = run_molpro_via_slurm(cfg.run_sh_path, rundir,
                                        sbatch_cmd=cfg.sbatch_cmd,
                                        poll_s=cfg.poll_s,
                                        max_wait_s=cfg.max_wait_s)


    metrics = parse_metrics(out_text, phase=phase)
    if metrics.get("converged"):
        seed_metrics = load_or_save_seed_metrics(cfg, metrics, logger_file)
    #if init_run and metrics["converged"] == True:
    #    init_run = False
    #    seed_metrics = metrics
        #logger.info("Initial run converged. Metrics: %s", seed_metrics)
    normalised_metrics = get_normalised_metrics(seed_metrics, metrics)
    logger_file.info("Normalised metrics: %s", ", ".join(f"{k}={v:.4f}" for k, v in normalised_metrics.items() if v is not None))
    sc = score_from_metrics(exps ,normalised_metrics, cfg.weights, cfg.s_ovrlp_penalty,cfg.redundancy_penalty, cfg.a_coupling_penalty,phase=phase)
    
    if phase == "log":
        logger_file.info("The score which is considered is %s", sc)
    with open(rundir / "metrics.json", "w") as f:
        json.dump(
            {"mode": cfg.mode, "theta": list(map(float, theta)), "exponents": list(map(float, exps)),
             "metrics": metrics, "score": sc},
            f, indent=2
        )
    return sc
