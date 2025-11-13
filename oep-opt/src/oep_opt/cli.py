# Placeholder for cli.py. Full code was in assistant's answer.
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import sys
import logging

from .logging_setup import setup_logging
from .config import JobConfig, Weights
from .workflow import objective, exps_from_theta
from .parameterizations import ensure_descending
from .utils import read_exps_from_file

def main(argv=None):
    p = argparse.ArgumentParser(description="Optimize OEP aux s-exponents (even-tempered or free) using SLURM run.sh per trial.")
    # System
    p.add_argument("--elem", default="O")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--orbital-parent", default="aug-cc-pwCV5Z")
    p.add_argument("--aux-parent", default="aug-cc-pVDZ/mp2fit")
    p.add_argument("--dm-file", default="dm.dat")
    p.add_argument("--e-ref", type=float, default=-75.056798837342)

    # Template + work
    p.add_argument("--template", default="molpro_template.inp")
    p.add_argument("--workdir", default="runs_O")

    # Metrics weights
    p.add_argument("--w-dvext", type=float, default=3.0)
    p.add_argument("--w-du", type=float, default=1.0)
    p.add_argument("--w-lieb", type=float, default=1.0)
    p.add_argument("--w-norm", type=float, default=1.0)
    p.add_argument("--w-rscaled-norm", type=float, default=1.0)
    p.add_argument("--w-sqrtrscaled-norm", type=float, default=1.0)
    p.add_argument("--w-rtimes-scaled-norm", type=float, default=1.0)


    # SLURM
    p.add_argument("--run-sh", required=True)
    p.add_argument("--sbatch-cmd", default="sbatch")
    p.add_argument("--poll-s", type=int, default=20)
    p.add_argument("--max-wait-s", type=int, default=0)

    # Optimization
    p.add_argument("--mode", choices=["even_tempered", "free_exponents"], default="even_tempered")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--order-penalty", type=float, default=0.0)

    # Even-tempered seed / fallback seed
    p.add_argument("--init-alpha-hi", type=float, default=100.0)
    p.add_argument("--init-beta", type=float, default=2.5)

    # Free-exponents initial values
    p.add_argument("--init-exps", type=str, default=None)
    p.add_argument("--init-exps-file", type=str, default=None)

    # SciPy minimize
    p.add_argument("--method", type=str, default="Powell")
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--gtol", type=float, default=1e-6)
    p.add_argument("--eps", type=float, default=1e-3)

    p.add_argument("command", choices=["optimize"], nargs="?", default="optimize")
    args = p.parse_args(argv)



    template_text = Path(args.template).read_text()
    weights = Weights(w_dvext=args.w_dvext, w_du=args.w_du, w_lieb=args.w_lieb, w_norm=args.w_norm, w_rscaled_norm=args.w_rscaled_norm, w_sqrtrscaled_norm=args.w_sqrtrscaled_norm, w_rtimes_scaled_norm=args.w_rtimes_scaled_norm)

    cfg = JobConfig(
        elem=args.elem, charge=args.charge, spin=args.spin,
        orbital_parent=args.orbital_parent, aux_parent=args.aux_parent,
        dm_file=args.dm_file, e_ref=args.e_ref,
        template_text=template_text, workroot=Path(args.workdir), run_sh_path=Path(args.run_sh),
        mode=args.mode, K=args.K, weights=weights,
        sbatch_cmd=args.sbatch_cmd, poll_s=args.poll_s, max_wait_s=args.max_wait_s,
        exp_min=1e-6, exp_max=1e6, order_penalty=float(args.order_penalty),
    )

    rundir = cfg.workroot
    rundir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path(rundir) / "run.log")

    logger.info("Starting OEP optimization for %s (mode=%s, K=%d)", cfg.elem, cfg.mode, cfg.K)
    logger.info("Minimization parameters: method=%s, maxiter=%d, gtol=%.3e, eps=%.3e",
                args.method, args.maxiter, args.gtol, args.eps)
    logger.info("Initial weights: w_dvext = %.3f, w_du = %.3f, w_dlieb = %.3f,w_dnorm = %.3f, w_rscaled_dnorm = %.3f, w_sqrtrscaled_norm = %.3f, w_rtimes_scaled_norm = %.3f",
                args.w_dvext, args.w_du, args.w_lieb, args.w_norm, args.w_rscaled_norm, args.w_sqrtrscaled_norm, args.w_rtimes_scaled_norm)
    # Build x0
    if args.mode == "even_tempered":
        # Parse & clamp inputs
        alpha_hi = float(args.init_alpha_hi)
        beta_in  = float(args.init_beta)
        beta     = max(1.0 + 1e-8, beta_in)  # ensure strictly > 1

        # Parametrization
        u = float(np.log(alpha_hi))
        v = float(np.log(beta - 1.0))

        x0 = np.array([u, v], dtype=float)

        # Construct initial seed exponents (descending): e_k = alpha_hi * beta^{-k}
        # Requires cfg.K and cfg.elem to be in scope
        seed = [alpha_hi / (beta ** k) for k in range(cfg.K)]

        #logger.info("Starting even-tempered optimization for %s with K=%d", cfg.elem, cfg.K)
        logger.info(
            "Initial guess: alpha_hi=%.6g, beta=%.6g  (u=%.6f, v=%.6f)",
            alpha_hi, beta, u, v
        )
        logger.info("Seed exponents: %s", ", ".join(f"{e:.6f}" for e in seed))
    else:
        if args.init_exps is None and args.init_exps_file is None:
            alpha0 = float(args.init_alpha_hi)
            beta0 = float(args.init_beta)
            seed = [alpha0 * (beta0 ** (-k)) for k in range(cfg.K)]
            seed = ensure_descending(seed)
        else:
            if args.init_exps:
                seed = [float(t) for t in args.init_exps.replace(",", " ").split()]
            else:
                seed = read_exps_from_file(Path(args.init_exps_file), args.elem)
        if len(seed) != cfg.K:
            raise ValueError(f"Provided {len(seed)} exponents, but K={cfg.K}.")
        x0 = np.log(np.array(seed, dtype=float))
        #logger.info("Starting free_exponents optimization for %s with K=%d", cfg.elem, cfg.K)
        logger.info("Initial seed exponents: %s", ", ".join(f"{e:.6f}" for e in seed))
        #print(f"[INFO] Starting free_exponents optimization for {cfg.elem} with K={cfg.K}")
        #print(f"[INFO] Initial seed exponents: {', '.join(f'{e:.4g}' for e in seed)}")
    #sys.stdout.flush()
    # Optimize
    res = minimize(
        fun=objective, x0=x0, args=(cfg,), method=args.method,
        options={"maxiter": args.maxiter, "disp": True, "gtol": args.gtol, "eps": args.eps}
        if args.method in {"BFGS", "CG", "Newton-CG", "L-BFGS-B", "TNC"}
        else {"maxiter": args.maxiter, "disp": True}
    )
    
    logger.info("=== Optimization result ===\n%s", res)

    #print("\n=== Optimization result ===")
    #print(res)

    best_exps = exps_from_theta(np.array(res.x, dtype=float), cfg)
    best_exps = ensure_descending(best_exps)
    snippet = f"s,{cfg.elem}," + ", ".join(f"{e:.8g}" for e in best_exps) + ".\n"
    out_snip = Path(cfg.workroot) / f"optimized_s_{cfg.elem}_{cfg.mode}.bas"
    out_snip.write_text(snippet)
    logger.info("Saved optimized s-shell snippet -> %s", out_snip)
    #print(f"\n[OK] Saved optimized s-shell snippet -> {out_snip}")
