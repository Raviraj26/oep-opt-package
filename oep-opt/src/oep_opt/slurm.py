# Placeholder for slurm.py. Full code was in assistant's answer.
from pathlib import Path
import os, re, shutil, subprocess, time
from typing import Tuple

def submit_slurm_and_wait(run_dir: Path, sbatch_cmd: str, poll_s: int = 20,
                          max_wait_s: int = 0,
                          sentinel: str = "Molpro calculation terminated") -> Tuple[int, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "_submitted"
    out_file = run_dir / "output"
    err_file = run_dir / "error"

    if out_file.exists():
        try:
            txt = out_file.read_text(errors="ignore")
            if sentinel in txt:
                err_txt = err_file.read_text(errors="ignore") if err_file.exists() else ""
                return (0, txt + "\n" + err_txt)
        except Exception:
            pass

    if marker.exists():
        job_id = marker.read_text().strip()
    else:
        for p in run_dir.glob("slurm-*.out"):
            try: p.unlink()
            except Exception: pass
        proc = subprocess.run(sbatch_cmd.split() + ["run.sh"], cwd=str(run_dir),
                              capture_output=True, text=True, check=False)
        sb_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"Submitted batch job\s+(\d+)", sb_out) or re.search(r"(\d+)", sb_out)
        job_id = m.group(1) if m else "UNKNOWN"
        marker.write_text(str(job_id))

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
        try: out_text = out_file.read_text(errors="ignore") if out_file.exists() else ""
        except Exception: out_text = ""
        if out_text and ("Molpro calculation terminated" in out_text or dnorm_pat.search(out_text)):
            break
        time.sleep(2)

    try:
        out_text += "\n" + (err_file.read_text(errors="ignore") if err_file.exists() else "")
    except Exception:
        pass

    rc = 0
    if "ERROR" in out_text and "WARNING" not in out_text: rc = 1
    if "Molpro calculation terminated" not in out_text: rc = max(rc, 1)
    return (rc, out_text)

def run_molpro_via_slurm(run_sh_source: Path, workdir: Path,
                         sbatch_cmd: str = "sbatch", poll_s: int = 20, max_wait_s: int = 0) -> Tuple[int, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    run_sh_target = workdir / "run.sh"
    if (not run_sh_target.exists()) or (run_sh_source.read_bytes() != run_sh_target.read_bytes()):
        shutil.copy(run_sh_source, run_sh_target)
        os.chmod(run_sh_target, 0o755)
    #input_target = workdir / "input"
    #if inp_path.name != "input" or inp_path.parent != workdir:
    #    shutil.copy(inp_path, input_target)
    return submit_slurm_and_wait(workdir, sbatch_cmd, poll_s=poll_s, max_wait_s=max_wait_s,
                                 sentinel="Molpro calculation terminated")

