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
from .utils import stable_tag_from_theta

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
    exps_desc = exps

    s_line = format_exps_for_molpro(exps_desc)
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
            ", ".join(f"{e:.6f}" for e in exps_desc),
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



    rc, out_text = run_molpro_via_slurm(cfg.run_sh_path, rundir,
                                        sbatch_cmd=cfg.sbatch_cmd,
                                        poll_s=cfg.poll_s,
                                        max_wait_s=cfg.max_wait_s)

    metrics = parse_metrics(out_text, phase=phase)
    sc = score_from_metrics(exps_desc ,metrics, cfg.weights, cfg.s_ovrlp_penalty,cfg.redundancy_penalty, cfg.a_coupling_penalty,phase=phase)
    
    if cfg.order_penalty > 0.0:
        for i in range(1, len(exps_desc)):
            if exps_desc[i] >= exps_desc[i - 1]:
                sc += cfg.order_penalty
    if phase == "log":
        logger_file.info("The score which is considered is %s", sc)
    with open(rundir / "metrics.json", "w") as f:
        json.dump(
            {"mode": cfg.mode, "theta": list(map(float, theta)), "exponents": exps_desc,
             "metrics": metrics, "score": sc},
            f, indent=2
        )
    return sc
