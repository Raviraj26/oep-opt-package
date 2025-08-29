#!/usr/bin/env python3
"""
OEP aux-basis s-shell optimizer (SLURM + Molpro)
-------------------------------------------------

Professional, readable refactor of your previous script. Two optimization modes:

1) even_tempered  (legacy):
   - Parameterizes K s-exponents with (alpha_hi, beta) where beta > 1.
   - Good for compact search when you want geometric spacing.

2) free_exponents  (NEW):
   - Optimizes K individual s-exponents directly.
   - Designed to *start from* your AUX parent s-primitives (e.g. aug-cc-pVDZ/mp2fit)
     and refine them.
   - You can provide initial exponents via --init-exps or --init-exps-file.
   - If not provided, we can still seed from even_tempered and then switch to free_exponents.

Key features
- Clean dataclasses for config, strong typing, explicit docs.
- Robust SLURM orchestration: single submit per run dir + sentinel wait.
- Safe exponent transforms (positivity, optional order penalty).
- Clear logging + metrics capture in JSON, snippet export for Molpro basis line.

Usage examples
--------------
# A) Optimize existing AUX s exponents (recommended)
python oep_opt_refit_s_from_aux.py \
  --mode free_exponents \
  --elem O --aux-parent aug-cc-pVDZ/mp2fit \
  --init-exps "100.0,33.3333,11.1111,3.7037,1.2346,0.4115,0.1372,0.0457" \
  --run-sh /path/to/run.sh --template molpro_template.inp

# B) Same, but read initial exps from a snippet file (line like: "s,O,100.0,33.33,...")
python oep_opt_refit_s_from_aux.py --mode free_exponents \
  --init-exps-file my_aux_s_snippet.bas --elem O \
  --run-sh /path/to/run.sh --template molpro_template.inp

# C) Legacy even-tempered search
python oep_opt_refit_s_from_aux.py --mode even_tempered \
  --K 10 --init-alpha-hi 100.0 --init-beta 2.5 \
  --run-sh /path/to/run.sh --template molpro_template.inp
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import logging
import sys
logger = logging.getLogger("oep-opt")
import numpy as np

try:
    from scipy.optimize import minimize
except Exception as e:  # pragma: no cover
    print("ERROR: SciPy is required. Install with: pip install scipy", file=sys.stderr)
    raise

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logging(logfile=None, level=logging.INFO):
    logger = logging.getLogger("oep-opt")
    logger.setLevel(level)

    # clear handlers if re-called
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console (stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.flush = sys.stdout.flush  # ensures auto-flush
    logger.addHandler(sh)

    # optional file handler
    if logfile:
        fh = logging.FileHandler(logfile, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus for positivity mapping."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _ensure_descending(exps: Sequence[float]) -> List[float]:
    """Return a strictly descending version of exps (ties nudged)."""
    out = list(sorted(exps, reverse=True))
    for i in range(1, len(out)):
        if out[i] >= out[i - 1]:
            out[i] = max(1e-12, out[i - 1] * 0.9999)
    return out


# -----------------------------------------------------------------------------
# Basis exponent parameterizations
# -----------------------------------------------------------------------------

def even_tempered_from_params(theta: np.ndarray, K: int,
                              exp_min: float, exp_max: float) -> List[float]:
    """Map (u, v) -> K even-tempered exponents.

    Parameters
    ----------
    theta : array_like of shape (2,)
        Unconstrained variables. u ~ log alpha_hi, v ~ log(beta-1).
    K : int
        Number of primitives.
    exp_min, exp_max : float
        Safety clipping range.

    Returns
    -------
    List[float]
        Strictly descending, clipped exponents.
    """
    u, v = float(theta[0]), float(theta[1])
    alpha_hi = float(_softplus(np.array([u]))[0]) + 1e-8  # > 0
    beta = 1.0 + float(_softplus(np.array([v]))[0])       # > 1

    exps = [alpha_hi * (beta ** (-k)) for k in range(K)]  # descending ideally
    # Clip and enforce descending strictly
    exps = [min(max(x, exp_min), exp_max) for x in exps]
    exps = _ensure_descending(exps)
    return exps


def free_exponents_from_params(theta: np.ndarray, K: int,
                               exp_min: float, exp_max: float) -> List[float]:
    """Map length-K unconstrained vector to K positive exponents.

    We exponentiate to ensure positivity, then clip + enforce descending.
    """
    if len(theta) != K:
        raise ValueError(f"theta length {len(theta)} must equal K={K} in free_exponents mode.")
    # Positivity via exp; alternative: softplus. Here exp is fine; logs are more interpretable.
    exps = list(np.exp(theta))
    exps = [min(max(x, exp_min), exp_max) for x in exps]
    exps = _ensure_descending(exps)
    return exps


# -----------------------------------------------------------------------------
# Rendering & formatting
# -----------------------------------------------------------------------------

def format_exps_for_molpro(exps: Sequence[float]) -> str:
    """Molpro basis line content: "100.0, 33.3, ... ." (trailing dot required)."""
    return ", ".join(f"{x:.8g}" for x in exps) + "."


def render_input(template_text: str, elem: str, charge: int, spin: int,
                 orbital_parent: str, aux_parent: str, s_exps_line: str,
                 dm_file: str, e_ref: float) -> str:
    """Fill placeholders in the Molpro template."""
    out = template_text
    out = out.replace("@ELEM@", elem)
    out = out.replace("@CHARGE@", str(charge))
    out = out.replace("@SPIN@", str(spin))
    out = out.replace("@ORBITAL_PARENT@", orbital_parent)
    out = out.replace("@AUX_PARENT@", aux_parent)
    out = out.replace("@S_EXP_LINE@", s_exps_line)
    #out = out.replace("@DM_FILE@", dm_file)
    out = out.replace("@E_REF@", f"{e_ref:.12f}")
    return out

def stage_dm_as_link(dm_path: str | Path, rundir: Path) -> Path | None:
    dm_src = Path(dm_path)
    if not dm_src.exists():
        return None

    #rundir.mkdir(parents=True, exist_ok=True)
    dm_src = dm_src.resolve()  # absolute source
    tgt = rundir / dm_src.name

    # If something is already there, replace it (file or symlink)
    if tgt.exists() or tgt.is_symlink():
        tgt.unlink()

    # Prefer a relative symlink (portable if you move the whole rundir tree)
    try:
        rel = os.path.relpath(dm_src, start=rundir)
        tgt.symlink_to(rel)
        return tgt
    except OSError:
        # Symlink not allowed? try hardlink (same filesystem only)
        try:
            os.link(dm_src, tgt)
            return tgt
        except OSError:
            # Last resort: copy
            shutil.copy2(dm_src, tgt)
            return tgt

# -----------------------------------------------------------------------------
# SLURM helpers (single-submit + sentinel wait)
# -----------------------------------------------------------------------------

def submit_slurm_and_wait(
    run_dir: Path,
    sbatch_cmd: str,
    poll_s: int = 20,
    max_wait_s: int = 0,
    sentinel: str = "Molpro calculation terminated",
) -> Tuple[int, str]:
    """Submit run.sh exactly once and block until done.

    Returns (rc, combined_output). rc=0 on heuristic success.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "_submitted"
    out_file = run_dir / "output"
    err_file = run_dir / "error"

    # Respect completed jobs
    if out_file.exists():
        try:
            txt = out_file.read_text(errors="ignore")
            if sentinel in txt:
                err_txt = err_file.read_text(errors="ignore") if err_file.exists() else ""
                return (0, txt + "\n" + err_txt)
        except Exception:
            pass

    # Submit only once
    if marker.exists():
        job_id = marker.read_text().strip()
    else:
        for p in run_dir.glob("slurm-*.out"):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            proc = subprocess.run(
                sbatch_cmd.split() + ["run.sh"],
                cwd=str(run_dir), capture_output=True, text=True, check=False
            )
        except FileNotFoundError as e:
            return (997, f"SBATCH_NOT_FOUND: {e}")
        sb_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"Submitted batch job\s+(\d+)", sb_out) or re.search(r"(\d+)", sb_out)
        job_id = m.group(1) if m else "UNKNOWN"
        marker.write_text(str(job_id))

    # Poll for completion
    start = time.time()
    while True:
        if out_file.exists():
            try:
                txt = out_file.read_text(errors="ignore")
                if sentinel in txt:
                    time.sleep(5)
                    break
            except Exception:
                pass
        if job_id != "UNKNOWN":
            q = subprocess.run(["squeue", "-j", str(job_id), "-h"],
                               cwd=str(run_dir), capture_output=True, text=True)
            if q.returncode == 0 and q.stdout.strip() == "":
                time.sleep(5)
                break
        if max_wait_s > 0 and (time.time() - start) > max_wait_s:
            return (998, f"TIMEOUT_WAITING_FOR_SLURM (job {job_id})")
        time.sleep(poll_s)

    out_text = ""
    end_wait_start = time.time()
    dnorm_pat = re.compile(r"(?im)^\s*SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)")
    
    while (time.time() - end_wait_start) < 60:
        try:
            out_text = out_file.read_text(errors="ignore") if out_file.exists() else ""
        except Exception:
            out_text = ""
        if not out_text:
            time.sleep(2); continue
        if ("Molpro calculation terminated" in out_text) or dnorm_pat.search(out_text):
            break
        time.sleep(2)
    
    # Append error file if present (same as before)
    try:
        out_text += "\n" + (err_file.read_text(errors="ignore") if err_file.exists() else "")
    except Exception:
        pass
    
    rc = 0
    if "ERROR" in out_text and "WARNING" not in out_text:
        rc = 1
    if "Molpro calculation terminated" not in out_text:
        rc = max(rc, 1)
    return (rc, out_text)


def run_molpro_via_slurm(
    run_sh_source: Path,
    inp_path: Path,
    workdir: Path,
    sbatch_cmd: str = "sbatch",
    poll_s: int = 20,
    max_wait_s: int = 0,
) -> Tuple[int, str]:
    """Copy run.sh + input into workdir, submit once, then wait."""
    workdir.mkdir(parents=True, exist_ok=True)

    run_sh_target = workdir / "run.sh"
    if (not run_sh_target.exists()) or (run_sh_source.read_bytes() != run_sh_target.read_bytes()):
        shutil.copy(run_sh_source, run_sh_target)
        os.chmod(run_sh_target, 0o755)

    #input_target = workdir / "input"
    #if inp_path.name != "input" or inp_path.parent != workdir:
    #    shutil.copy(inp_path, input_target)

    return submit_slurm_and_wait(
        workdir, sbatch_cmd, poll_s=poll_s, max_wait_s=max_wait_s,
        sentinel="Molpro calculation terminated"
    )


# -----------------------------------------------------------------------------
# Metrics parsing & scoring
# -----------------------------------------------------------------------------
DEFAULT_PATTERNS: Dict[str, List[str]] = {
    "dvext": [r"(?m)^\s*KSINV\s+External\s+energy\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "du":    [r"(?m)^\s*KSINV\s+Hartree\s+energy\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "dlieb": [r"(?m)^\s*KSINV\s+Lieb\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    # From line: "Total Density Test       0.0115009353  ..."
    #"dnorm": [r"(?im)^\s*Total\s+Density\s+Test\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "dnorm": [r"(?im)^\s*SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "converged": [r"(?im)^\s*SCF\s+Converged\s"],
    "not_converged": [r"(?im)^\s*SCF\s+NOT\s+Converged\s"],
}


def _detect_conv(out_text: str, yes_pats: Iterable[str], no_pats: Iterable[str]) -> Optional[bool]:
    for pat in no_pats:
        if re.search(pat, out_text):
            return False
    for pat in yes_pats:
        if re.search(pat, out_text):
            return True
    return None


def _first_float(text: str, patterns: Sequence[str]) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def parse_metrics(out_text: str) -> Dict[str, Optional[float]]:
    dvext = _first_float(out_text, DEFAULT_PATTERNS["dvext"])
    du    = _first_float(out_text, DEFAULT_PATTERNS["du"])
    dlieb = _first_float(out_text, DEFAULT_PATTERNS["dlieb"])
    dnorm = _first_float(out_text, DEFAULT_PATTERNS["dnorm"])
    conv  = _detect_conv(out_text, DEFAULT_PATTERNS["converged"], DEFAULT_PATTERNS["not_converged"])
    logger.info("Parsed metrics: dvext=%s, du=%s, dlieb=%s, dnorm=%s, converged=%s", dvext, du, dlieb, dnorm, conv)
    #print(f"[PARSED METRICS]: dvext={dvext}, du={du}, dlieb={dlieb}, dnorm={dnorm}, converged={conv}")
    return {"dvext": dvext, "du": du, "dlieb": dlieb, "dnorm": dnorm, "converged": conv}


@dataclass
class Weights:
    w_dvext: float = 3.0
    w_du: float = 1.0
    w_lieb: float = 1.0
    w_norm: float = 1.0
    penalty_noconv: float = 1e4


def score_from_metrics(metrics: Dict[str, Optional[float]], weights: Weights, fail_penalty: float = 1e6) -> float:
    dv, du, dlieb, dnorm = (metrics.get(k) for k in ("dvext", "du", "dlieb", "dnorm"))
    terms: List[float] = []
    if dv is not None:    terms.append(weights.w_dvext * abs(dv))
    if du is not None:    terms.append(weights.w_du    * abs(du))
    if dlieb is not None: terms.append(weights.w_lieb  * abs(dlieb))
    if dnorm is not None: terms.append(weights.w_norm  * abs(dnorm))
    if not terms:
        return fail_penalty
    sc = sum(terms)
    logger.info("Score components: %s", ", ".join(f"{t:.12f}" for t in terms))
    logger.info("Total score: %.12f", sc)
    conv = metrics.get("converged")
    if conv is False:
        sc += weights.penalty_noconv
    elif conv is None:
        sc += 0.1 * weights.penalty_noconv
    if not np.isfinite(sc):
        return fail_penalty
    return sc


# -----------------------------------------------------------------------------
# Job configuration
# -----------------------------------------------------------------------------
@dataclass
class JobConfig:
    # Chemical/system
    elem: str
    charge: int
    spin: int
    orbital_parent: str
    aux_parent: str
    dm_file: str
    e_ref: float

    # Template & working paths
    template_text: str
    workroot: Path
    run_sh_path: Path

    # Optimization
    mode: str  # "even_tempered" | "free_exponents"
    K: int
    weights: Weights
    sbatch_cmd: str = "sbatch"
    poll_s: int = 20
    max_wait_s: int = 0
    exp_min: float = 1e-6
    exp_max: float = 1e6
    order_penalty: float = 0.0  # add to score if not strictly descending


# -----------------------------------------------------------------------------
# Input writing
# -----------------------------------------------------------------------------

def write_input_file(cfg: JobConfig, exps: Sequence[float], rundir: Path) -> Path:
    s_line = format_exps_for_molpro(exps)
    inp_text = render_input(
        cfg.template_text, cfg.elem, cfg.charge, cfg.spin,
        cfg.orbital_parent, cfg.aux_parent, s_line, cfg.dm_file, cfg.e_ref,
    )
    inp_path = rundir / "input"
    inp_path.write_text(inp_text)
    return inp_path


# -----------------------------------------------------------------------------
# Objective function
# -----------------------------------------------------------------------------

def _exps_from_theta(theta: np.ndarray, cfg: JobConfig) -> List[float]:
    if cfg.mode == "even_tempered":
        return even_tempered_from_params(theta, cfg.K, cfg.exp_min, cfg.exp_max)
    elif cfg.mode == "free_exponents":
        return free_exponents_from_params(theta, cfg.K, cfg.exp_min, cfg.exp_max)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


def objective(theta: np.ndarray, cfg: JobConfig) -> float:
    # Compact, stable run directory tag
    tag_bytes = json.dumps(list(map(float, theta))).encode("utf-8")
    tag = hashlib.sha1(tag_bytes).hexdigest()[:10]
    rundir = cfg.workroot / f"run_{cfg.elem}_{cfg.mode}_{tag}"
    rundir.mkdir(parents=True, exist_ok=True)

    # Stage dm.dat
#    dm_src = Path(cfg.dm_file)
#    if dm_src.exists():
#        tgt = rundir / dm_src.name
#        if not tgt.exists():
#            shutil.copy(dm_src, tgt)

    stage_dm_as_link(cfg.dm_file, rundir)

    # Map params -> exponents and write input
    exps = _exps_from_theta(theta, cfg)
    exps_desc = _ensure_descending(exps)

    inp_path = write_input_file(cfg, exps_desc, rundir)

    try:
        logger.info(
            " %s: mode=%s, K=%d: %s -> %s",
            cfg.elem,
            cfg.mode,
            cfg.K,
            ", ".join(f"{e:.6f}" for e in exps_desc),
            rundir.name,
        )
    except Exception:
        logger.info(
            " %s: mode=%s, K=%d -> %s",
            cfg.elem,
            cfg.mode,
            cfg.K,
            rundir.name,
        )

    # SLURM run
    rc, out_text = run_molpro_via_slurm(
        cfg.run_sh_path, inp_path, rundir,
        sbatch_cmd=cfg.sbatch_cmd, poll_s=cfg.poll_s, max_wait_s=cfg.max_wait_s,
    )
    #if rc != 0:
    #    return 1e9

    # Metrics -> scalar score
    #print(cfg
    metrics = parse_metrics(out_text)
    #print("metrics from the output",metrics)
    sc = score_from_metrics(metrics, cfg.weights)

    # Optional: penalize ordering violations (should be tiny or zero after _ensure_descending)
    if cfg.order_penalty > 0.0:
        for i in range(1, len(exps_desc)):
            if exps_desc[i] >= exps_desc[i - 1]:
                sc += cfg.order_penalty

    # Log
    #print("test1")
    with open(rundir / "metrics.json", "w") as f:
        json.dump(
            {
                "mode": cfg.mode,
                "theta": list(map(float, theta)),
                "exponents": exps_desc,
                "metrics": metrics,
                "score": sc,
            },
            f,
            indent=2,
        )
        #print("test2")
    sys.stdout.flush()
    return sc


# -----------------------------------------------------------------------------
# Initializers for theta
# -----------------------------------------------------------------------------

def parse_init_exps_str(s: str) -> List[float]:
    """Parse comma/space separated numbers into a list of floats."""
    toks = [t.strip() for t in re.split(r"[ ,]+", s.strip()) if t.strip()]
    return [float(t) for t in toks]


def parse_s_exps_from_molpro_table(text: str, center: int = 1) -> Optional[List[float]]:
    """Parse S-exponents from a Molpro basis table dump.

    Looks for lines like:
        Centre l       exponent    contractions
         1   s         109.461000  1.000000
    Returns a list of S exponents for the given centre, if found.
    """
    exps: List[float] = []
    in_table = False
    for raw in text.splitlines():
        line = raw.strip("\n")
        if not in_table:
            if ("Centre" in line and "exponent" in line and "contractions" in line):
                in_table = True
            continue
        if not line.strip():
            continue
        # Skip pure separator lines like "===="
        if set(line.strip()) == {"="}:
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].lower() == 's':
            try:
                idx = int(parts[0])
                if idx == center:
                    exps.append(float(parts[2]))
            except Exception:
                pass
    return exps or None


def read_exps_from_file(path: Path, elem: str) -> List[float]:
    """Read exponents from a simple snippet or a Molpro table dump.

    Accepted forms (first match wins):
      1) A Molpro line:   s,O, 100.0, 33.3, ... .
      2) A Molpro basis table block ("Centre l       exponent    contractions").
      3) Plain numbers on one line: 100.0, 33.3, ...
    """
    txt = path.read_text()
    # Try Molpro single-line snippet first
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith('s,'):
            parts = [p.strip().rstrip('.') for p in s.split(',')]
            if len(parts) >= 3 and parts[1].upper() == elem.upper():
                return [float(p) for p in parts[2:] if p]
    # Try Molpro table dump
    table_exps = parse_s_exps_from_molpro_table(txt, center=1)
    if table_exps:
        return table_exps
    # Fallback: first line of comma/space-separated numbers
    for line in txt.splitlines():
        s = line.strip().rstrip('.')
        if not s:
            continue
        toks = [t for t in s.replace(',', ' ').split() if t]
        nums: List[float] = []
        ok = True
        for t in toks:
            try:
                nums.append(float(t))
            except Exception:
                ok = False
                break
        if ok and nums:
            return nums
    raise ValueError(f"Could not parse exponents from {path}")


def make_initial_theta(args: argparse.Namespace, mode: str, K: int): 
    if mode == "even_tempered":
        # theta = [u, v] unconstrained; u ~ log alpha_hi, v ~ log(beta-1)
        u = float(np.log(float(args.init_alpha_hi)))
        v = float(np.log(max(1e-8, float(args.init_beta) - 1.0)))
        return np.array([u, v], dtype=float)
    elif mode == "free_exponents":
        if args.init_exps is None and args.init_exps_file is None:
            # Fall back to a gentle geometric seed
            alpha0 = float(args.init_alpha_hi)
            beta0 = float(args.init_beta)
            seed = [alpha0 * (beta0 ** (-k)) for k in range(K)]
            seed = _ensure_descending(seed)
        else:
            if args.init_exps:
                seed = parse_init_exps_str(args.init_exps)
            else:
                seed = read_exps_from_file(Path(args.init_exps_file), args.elem)
        if len(seed) != K:
            raise ValueError(f"Provided {len(seed)} exponents, but K={K}.")
        # theta are logs for positivity
        return np.log(np.array(seed, dtype=float))
    else:
        raise ValueError(f"Unknown mode {mode}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Optimize OEP aux s-exponents (even-tempered or free) using SLURM run.sh per trial.")

    # System
    p.add_argument("--elem", default="O", help="Element symbol (e.g., O, H, N)")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=2)
    p.add_argument("--orbital-parent", default="aug-cc-pwCV5Z")
    p.add_argument("--aux-parent", default="aug-cc-pVDZ/mp2fit")
    p.add_argument("--dm-file", default="dm.dat", help="Path to reference density matrix (dm.dat)")
    p.add_argument("--e-ref", type=float, default=-75.056798837342, help="Reference energy for uksinv")

    # Template + work
    p.add_argument("--template", default="molpro_template.inp", help="Path to Molpro input template")
    p.add_argument("--workdir", default="runs_O", help="Directory for run folders")

    # Metrics weights
    p.add_argument("--w-dvext", type=float, default=3.0)
    p.add_argument("--w-du", type=float, default=1.0)
    p.add_argument("--w-lieb", type=float, default=1.0)
    p.add_argument("--w-norm", type=float, default=1.0)

    # SLURM
    p.add_argument("--run-sh", required=True, help="Path to your SLURM run.sh, copied into each run dir")
    p.add_argument("--sbatch-cmd", default="sbatch")
    p.add_argument("--poll-s", type=int, default=20)
    p.add_argument("--max-wait-s", type=int, default=0)

    # Optimization mode
    p.add_argument("--mode", choices=["even_tempered", "free_exponents"], default="free_exponents",
                   help="Search over geometric (2 params) or K free exponents")
    p.add_argument("--K", type=int, default=8, help="Number of s primitives")
    p.add_argument("--order-penalty", type=float, default=0.0, help="Penalty added if order not strictly descending")

    # Even-tempered seed/params (also used to make fallback seed for free_exponents)
    p.add_argument("--init-alpha-hi", type=float, default=100.0, help="Initial tightest exponent (alpha_hi) for seeds")
    p.add_argument("--init-beta", type=float, default=2.5, help="Initial geometric ratio beta (>1) for seeds")

    # Free-exponents initial values
    p.add_argument("--init-exps", type=str, default=None, help="Comma/space separated exponents to seed from (K numbers)")
    p.add_argument("--init-exps-file", type=str, default=None, help="File with initial s exponents: Molpro line 's,O,... .', OR Molpro basis table dump ('Centre l exponent contractions'), OR a single line of numbers.")

    # Optimizer
    p.add_argument("--method", type=str, default="Powell", help="scipy.optimize.minimize method (Powell|Nelder-Mead|BFGS|...) ")
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--gtol", type=float, default=1e-6)
    p.add_argument("--eps", type=float, default=1e-3)

    args = p.parse_args()

    template_text = Path(args.template).read_text()
    weights = Weights(w_dvext=args.w_dvext, w_du=args.w_du, w_lieb=args.w_lieb, w_norm=args.w_norm)

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

    # Determine initial theta
    x0 = make_initial_theta(args, cfg.mode, cfg.K)

    #print(f"[INFO] Starting {cfg.mode} optimization for {cfg.elem} with K={cfg.K}")
    if cfg.mode == "even_tempered":
        seed = [args.init_alpha_hi / (args.init_beta ** k) for k in range(cfg.K)]
        logger.info("Initial seed exponents: %s", ", ".join(f"{e:.6f}" for e in seed))
        #print(f"[INFO] Initial (seed) alpha_hi={args.init_alpha_hi:.6g}, beta={args.init_beta:.6g}")
    else:
        seed_exps = list(np.exp(x0))
        logger.info("Initial seed exponents: %s", ", ".join(f"{e:.6f}" for e in seed_exps))
        #print(f"[INFO] Initial seed exponents (K={cfg.K}): {', '.join(f'{e:.6f}' for e in seed_exps)}")

    sys.stdout.flush()
    # Optimize
    res = minimize(
        fun=objective, x0=x0, args=(cfg,), method=args.method,
        options={"maxiter": args.maxiter, "disp": True, "gtol": args.gtol, "eps": args.eps}
        if args.method in {"BFGS", "CG", "Newton-CG", "L-BFGS-B", "TNC"}
        else {"maxiter": args.maxiter, "disp": True}
    )

    print("\n=== Optimization result ===")
    print(res)

    # Final exponents + snippet
    best_exps = _exps_from_theta(np.array(res.x, dtype=float), cfg)
    best_exps = _ensure_descending(best_exps)

    snippet = f"s,{cfg.elem}," + ", ".join(f"{e:.8g}" for e in best_exps) + ".\n"
    out_snip = Path(cfg.workroot) / f"optimized_s_{cfg.elem}_{cfg.mode}.bas"
    out_snip.write_text(snippet)
    print(f"\n[OK] Saved optimized s-shell snippet -> {out_snip}")


if __name__ == "__main__":
    main()
