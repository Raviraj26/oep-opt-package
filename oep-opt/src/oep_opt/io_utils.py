# Placeholder for io_utils.py. Full code was in assistant's answer.
from __future__ import annotations
from pathlib import Path
from typing import Sequence, List, Optional, Tuple, Union
import os, shutil

import re
import datetime as dt


def format_exps_for_molpro(exps: Sequence[float]) -> str:
    return ", ".join(f"{x:.8g}" for x in exps) + "."

def render_input(template_text: str, elem: str, charge: int, spin: int,
                    alpha_occ: int, beta_occ: int, r_dnormcutoff: float,
                 orbital_parent: str, aux_parent: str, s_exps_line: str,
                 dm_file: str, e_ref: float) -> str:
    out = template_text
    out = out.replace("@ELEM@", elem)
    out = out.replace("@CHARGE@", str(charge))
    out = out.replace("@SPIN@", str(spin))
    out = out.replace("@ORBITAL_PARENT@", orbital_parent)
    out = out.replace("@AUX_PARENT@", aux_parent)
    out = out.replace("@S_EXP_LINE@", s_exps_line)
    out = out.replace("@E_REF@", f"{e_ref:.12f}")
    out = out.replace("@ALPHA_OCC@", str(alpha_occ))
    out = out.replace("@BETA_OCC@", str(beta_occ))
    out = out.replace("@R_DNORMCUTOFF@", str(r_dnormcutoff))
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


def write_input_file(template_text: str, elem: str, charge: int, spin: int,
                        alpha_occ: int, beta_occ: int, r_dnormcutoff: float,
                     orbital_parent: str, aux_parent: str, s_exps_line: str,
                     dm_file: str, e_ref: float, rundir: Path) -> Path:
    inp_text = render_input(template_text, elem, charge, spin, alpha_occ, beta_occ, r_dnormcutoff,
                            orbital_parent, aux_parent, s_exps_line, dm_file, e_ref)
    inp_path = rundir / "input"
    inp_path.write_text(inp_text)
    return inp_path


def _now_stamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _fmt_exps(exps: List[float]) -> str:
    return ", ".join(f"{float(x):.10g}" for x in exps)

def write_cases_output_with_best_exps(
    case_in_path: Path,
    case_out_path: Path,
    best_exps: Optional[Union[List[float], tuple]],
    ) -> Tuple[bool, str]:
    """
    Replace INIT_EXPS in a .env case file with best_exps and write old INIT_EXPS
    as a comment on the next line.

    Accepts lines like:
      INIT_EXPS="..."
      export INIT_EXPS="..."
      INIT_EXPS=...
    """

    if not best_exps:
        return False, "best_exps missing/empty (optimization likely failed)"

    try:
        lines = case_in_path.read_text(errors="ignore").splitlines(True)  # keep newlines
    except Exception as e:
        return False, f"could not read case file: {e}"

    # Capture:
    #  group(1): optional "export "
    #  group(2): RHS value (everything after '=' up to optional inline comment)
    #  group(3): optional inline comment (including '# ...')
    pat = re.compile(r'^(\s*(?:export\s+)?)INIT_EXPS\s*=\s*(.*?)\s*(#.*)?\s*$')

    init_idx = None
    old_rhs = None
    prefix = ""
    inline_comment = ""

    for i, line in enumerate(lines):
        m = pat.match(line.rstrip("\n"))
        if m:
            init_idx = i
            prefix = m.group(1) or ""
            old_rhs = (m.group(2) or "").strip()
            inline_comment = m.group(3) or ""
            break

    if init_idx is None:
        return False, "INIT_EXPS line not found (maybe different key name?)"

    # Strip wrapping quotes only for storing in comment
    old_clean = old_rhs
    if (old_clean.startswith('"') and old_clean.endswith('"')) or (old_clean.startswith("'") and old_clean.endswith("'")):
        old_clean = old_clean[1:-1]

    new_exps_str = _fmt_exps(list(best_exps))
    new_line = f'{prefix}INIT_EXPS="{new_exps_str}"{(" " + inline_comment) if inline_comment else ""}\n'
    comment = f'# PREV_INIT_EXPS="{old_clean}" UPDATED_AT={_now_stamp()}\n'

    lines[init_idx] = new_line

    # replace an existing prev line if it is already right below
    #if init_idx + 1 < len(lines) and lines[init_idx + 1].lstrip().startswith("# PREV_INIT_EXPS="):
    #    lines[init_idx + 1] = comment
    #else:
    lines.insert(init_idx + 1, comment)

    case_out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = case_out_path.with_suffix(case_out_path.suffix + ".tmp")
    tmp.write_text("".join(lines))
    os.replace(tmp, case_out_path)

    return True, "written"