# Placeholder for cli.py. Full code was in assistant's answer.
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import sys
import logging

from .logging_setup import setup_logging
from .config import JobConfig, Weights, S_ovrlp_penalty, Redundancy_penalty, A_coupling_penalty
from .workflow import objective, exps_from_theta
from .parameterizations import ensure_descending
from .utils import read_exps_from_file
from .concurrency import Evaluator, jac_central_parallel, jac_forward_parallel
from .io_utils import write_cases_output_with_best_exps

def str2bool(x):
    return x.lower() in ("yes","true","t","1","y")


def main(argv=None):
    p = argparse.ArgumentParser(description="Optimize OEP aux s-exponents (even-tempered or free) using SLURM run.sh per trial.")
    # System
    p.add_argument("--elem", default="O")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--orbital-parent", default="aug-cc-pwCV5Z")
    p.add_argument("--aux-parent", default="aug-cc-pVDZ/mp2fit")
    p.add_argument("--dm-file", default="dm.dat")
    p.add_argument("--input-case-file", default="elem.env")
    p.add_argument("--e-ref", type=float, default=-75.056798837342)
    p.add_argument("--alpha-occ", type=int, default=5)
    p.add_argument("--beta-occ", type=int, default=3)
    p.add_argument("--r-dnorm-cutoff", type=float, default=6.0)
    # Template + work
    p.add_argument("--template", default="molpro_template.inp")
    p.add_argument("--workdir", default="runs_O")

    # Metrics weights
    p.add_argument("--w-dvext", type=float, default=3.0)
    p.add_argument("--w-du", type=float, default=1.0)
    p.add_argument("--w-lieb", type=float, default=1.0)
    p.add_argument("--w-norm", type=float, default=1.0)
    p.add_argument("--w-rscaled-norm", type=float, default=1.0)
    p.add_argument("--w-rsqr-scaled-norm", type=float, default=1.0)
    p.add_argument("--w-sqrtrscaled-norm", type=float, default=1.0)
    p.add_argument("--w-rtimes-scaled-norm", type=float, default=1.0)
    
    ## S_overlap_penalty
    p.add_argument("--s_ovrlp_penalty_expo", type = float, default = 9.0)
    p.add_argument("--s_ovrlp_penalty_coeff", type = float, default = 1e-3)
    p.add_argument("--knob_for_s_ovrlp_penalty", type = str2bool, default = False)

    p.add_argument("--a_coupling_penalty_expo", type = float, default = 2.0)
    p.add_argument("--a_coupling_penalty_coeff", type = float, default = 1e-4)
    p.add_argument("--knob_for_a_coupling_penalty", type = str2bool, default = False)
    ## Redundancy Penalty for s exponents
    p.add_argument("--redundancy_penalty_coeff_a", type = float, default = 1.1)
    p.add_argument("--redundancy_penalty_coeff_b", type = float, default = 10.0)
    p.add_argument("--redundancy_penalty_coeff_c", type = float, default = 1e-3)
    p.add_argument("--redundancy_penalty_knob", type=str2bool, default = False)


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
    p.add_argument("--parallel_eval", type=str2bool, default=False)
    p.add_argument("--parallel_eval_workers", type=int, default=4)
    p.add_argument("--parallel_eval_method", choices=["central", "forward"], default="central")
    #p.add_argument("command", choices=["optimize"], nargs="?", default="optimize")
    args = p.parse_args(argv)



    template_text = Path(args.template).read_text()
    input_case_file = Path(args.input_case_file).resolve()  # resolve to absolute path for later use
    weights = Weights(w_dvext=args.w_dvext, w_du=args.w_du, w_lieb=args.w_lieb, w_norm=args.w_norm,
                      w_rscaled_norm=args.w_rscaled_norm, w_sqrtrscaled_norm=args.w_sqrtrscaled_norm,
                      w_rtimes_scaled_norm=args.w_rtimes_scaled_norm, w_rsqr_scaled_norm= args.w_rsqr_scaled_norm)
    s_ovrlp_penalty = S_ovrlp_penalty(coeff=args.s_ovrlp_penalty_coeff, expo = args.s_ovrlp_penalty_expo, knob = args.knob_for_s_ovrlp_penalty)
    a_coupling_penalty = A_coupling_penalty(coeff = args.a_coupling_penalty_coeff, expo = args.a_coupling_penalty_expo, knob = args.knob_for_a_coupling_penalty)
    
    redundancy_penalty = Redundancy_penalty(a = args.redundancy_penalty_coeff_a, b = args.redundancy_penalty_coeff_b, c = args.redundancy_penalty_coeff_c, knob = args.redundancy_penalty_knob)
    cfg = JobConfig(
        elem=args.elem, charge=args.charge, spin=args.spin,
        alpha_occ=args.alpha_occ, beta_occ=args.beta_occ, r_dnormcutoff=args.r_dnorm_cutoff,
        orbital_parent=args.orbital_parent, aux_parent=args.aux_parent,
        dm_file=args.dm_file, input_case_file=input_case_file, e_ref=args.e_ref,
        template_text=template_text, workroot=Path(args.workdir), run_sh_path=Path(args.run_sh),
        mode=args.mode, K=args.K, weights=weights, s_ovrlp_penalty=s_ovrlp_penalty,redundancy_penalty=redundancy_penalty,
        a_coupling_penalty = a_coupling_penalty, sbatch_cmd=args.sbatch_cmd, poll_s=args.poll_s, max_wait_s=args.max_wait_s,
        exp_min=1e-6, exp_max=1e6, order_penalty=float(args.order_penalty), logging=True
    )

    rundir = cfg.workroot
    rundir.mkdir(parents=True, exist_ok=True)
#    logger = setup_logging(Path(rundir) / "run.log")
    logger = setup_logging(run_logfile=Path(rundir) / "run.log", grad_logfile=Path(rundir) / "grad.log",)


    logger.info("Starting OEP optimization for %s (mode=%s, K=%d)", cfg.elem, cfg.mode, cfg.K)
    logger.info("Minimization parameters: method=%s, maxiter=%d, gtol=%.3e, eps=%.3e",
                args.method, args.maxiter, args.gtol, args.eps)
    logger.info("S_overlap penalty parameters expo=%s, coeff=%s, knob=%s", s_ovrlp_penalty.expo, s_ovrlp_penalty.coeff, s_ovrlp_penalty.knob)
    logger.info("A_coupling penalty parameters expo=%s, coeff=%s, knob=%s", a_coupling_penalty.expo, a_coupling_penalty.coeff, a_coupling_penalty.knob)
    logger.info("Redundancy penalty parameters a=%s,b=%s,c=%s, knob=%s",redundancy_penalty.a, redundancy_penalty.b, redundancy_penalty.c, redundancy_penalty.knob)
    logger.info("Initial weights: w_dvext = %.3f, w_du = %.3f, w_dlieb = %.3f,w_dnorm = %.3f, w_rscaled_dnorm = %.3f, w_sqrtrscaled_norm = %.3f, w_rtimes_scaled_norm = %.3f, w_rsqr_scaled_norm = %.3f", 
                args.w_dvext, args.w_du, args.w_lieb, args.w_norm, args.w_rscaled_norm, args.w_sqrtrscaled_norm, args.w_rtimes_scaled_norm, args.w_rsqr_scaled_norm)
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
    if args.parallel_eval:
        logger.info("Using parallel evaluation with %d workers, method=%s", args.parallel_eval_workers, args.parallel_eval_method)
        evaluator = Evaluator(cfg, max_workers=args.parallel_eval_workers)   # or a CLI arg like --fd-par

        def fun_wrapped(t):
            return evaluator.eval_one(t,  phase="log")

        def jac_wrapped(t):
            if args.parallel_eval_method == "forward":
                return jac_forward_parallel(t, evaluator, eps=args.eps)
            elif args.parallel_eval_method == "central":
                return jac_central_parallel(t, evaluator, eps=args.eps)

        res = minimize(
            fun=fun_wrapped,
            x0=x0,
            method="BFGS",
            jac=jac_wrapped,
            options={"maxiter": args.maxiter, "disp": True, "gtol": args.gtol, "eps": args.eps}
            if args.method in {"BFGS", "CG", "Newton-CG", "L-BFGS-B", "TNC"}
            else {"maxiter": args.maxiter, "disp": True}
            )
    else:
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

    # case file you started from (or the one in cases/)
    case_in = Path(cfg.input_case_file)          # you need to have this in cfg, or pass it around
    case_out = cfg.workroot / "case_output.env"  # or a central cases_output/<ELEM>.env

    ok = write_cases_output_with_best_exps(case_in, case_out, best_exps)
    logger.info("Saved case output with best exponents -> %s (success=%s)", case_out, ok)

    #print(f"\n[OK] Saved optimized s-shell snippet -> {out_snip}")
