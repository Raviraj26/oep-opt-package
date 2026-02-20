# Placeholder for io_utils.py. Full code was in assistant's answer.
from pathlib import Path
from typing import Sequence
import os, shutil
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
