import re
from typing import Dict, Iterable, Optional, Sequence, List
import logging
#logger = logging.getLogger("oep-opt")


#DEFAULT_PATTERNS: Dict[str, list[str]] = {
#    "dvext": [r"(?m)^\s*KSINV\s+External\s+energy\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
#    "du":    [r"(?m)^\s*KSINV\s+Hartree\s+energy\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
#    "dlieb": [r"(?m)^\s*KSINV\s+Lieb\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
#    "dnorm": [r"(?im)^\s*SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
#    "rscaled_dnorm": [r"(?im)^\s*R_SCALED_SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
#    "sqrtrscaled_dnorm": [r"(?im)^\s*SQRTR_SCALED_SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
#    "rtimes_scaled_dnorm": [r"(?im)^\s*Rtimes_SCALED_SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
#    "rsqr_scaled_dnorm": [r"(?im)^\s*RSQR_SCALED_SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
#    "converged": [r"(?im)^\s*SCF\s+Converged\s"],
#    "not_converged": [r"(?im)^\s*SCF\s+NOT\s+Converged\s"],
#    "s_ovrlp": [r"(?is)Eigenvalues of S\^I-matrix.*?\n[-\s]*\n\s*1\s+(?:\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
#    "a_matrix": [r"(?is)Eigenvalues of A\^\{III\}-matrix.*?\n[-\s]*\n\s*1\s+(?:\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
#}
DEFAULT_PATTERNS_L1: Dict[str, list[str]] = {
    "dvext": [r"(?m)^\s*KSINV\s+External\s+energy\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "du":    [r"(?m)^\s*KSINV\s+Hartree\s+energy\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "dlieb": [r"(?m)^\s*KSINV\s+Lieb\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+dnorm\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "rscaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+Without_R_cutoff_SYMMETRIZED\s+rscaled\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "sqrtrscaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+Without_R_cutoff_SYMMETRIZED\s+sqrtrscaled\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "rtimes_scaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+Without_R_cutoff_SYMMETRIZED\s+rtimes\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "rsqr_scaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+Without_R_cutoff_SYMMETRIZED\s+rsqrscaled\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "ref_proj_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+for\s+ref_proj_den_oep\s+dnorm\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "ref_proj_rscaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+for\s+ref_proj_den_oep\s+Without_R_cutoff_SYMMETRIZED\s+rscaled\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "ref_proj_sqrtrscaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+for\s+ref_proj_den_oep\s+Without_R_cutoff_SYMMETRIZED\s+sqrtrscaled\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "ref_proj_rtimes_scaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+for\s+ref_proj_den_oep\s+Without_R_cutoff_SYMMETRIZED\s+rtimes\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "ref_proj_rsqr_scaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+for\s+ref_proj_den_oep\s+Without_R_cutoff_SYMMETRIZED\s+rsqrscaled\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "converged": [r"(?im)^\s*SCF\s+Converged\s"],
    "not_converged": [r"(?im)^\s*SCF\s+NOT\s+Converged\s"],
    "s_ovrlp": [r"(?is)Eigenvalues of S\^I-matrix.*?\n[-\s]*\n\s*1\s+(?:\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "a_matrix": [r"(?is)Eigenvalues of A\^\{III\}-matrix.*?\n[-\s]*\n\s*1\s+(?:\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
}
DEFAULT_PATTERNS_L2: Dict[str, list[str]] = {
    "dvext": [r"(?m)^\s*KSINV\s+External\s+energy\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "du":    [r"(?m)^\s*KSINV\s+Hartree\s+energy\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "dlieb": [r"(?m)^\s*KSINV\s+Lieb\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+using\s+sqr\s+norm\s+dnorm\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "rscaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+using\s+sqr\s+norm\s+Without_R_cutoff_SYMMETRIZED\s+rscaled\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "sqrtrscaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+using\s+sqr\s+norm\s+Without_R_cutoff_SYMMETRIZED\s+sqrtrscaled\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "rtimes_scaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+using\s+sqr\s+norm\s+Without_R_cutoff_SYMMETRIZED\s+rtimes\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "rsqr_scaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+using\s+sqr\s+norm\s+Without_R_cutoff_SYMMETRIZED\s+rsqrscaled\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "ref_proj_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+using\s+sqr\s+norm\s+for\s+ref_proj_den_oep\s+dnorm\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "ref_proj_rscaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+using\s+sqr\s+norm\s+for\s+ref_proj_den_oep\s+Without_R_cutoff_SYMMETRIZED\s+rscaled\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "ref_proj_sqrtrscaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+using\s+sqr\s+norm\s+for\s+ref_proj_den_oep\s+Without_R_cutoff_SYMMETRIZED\s+sqrtrscaled\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "ref_proj_rtimes_scaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+using\s+sqr\s+norm\s+for\s+ref_proj_den_oep\s+Without_R_cutoff_SYMMETRIZED\s+rtimes\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "ref_proj_rsqr_scaled_dnorm": [r"(?im)^\s*radial\s+86\s+directions\s+using\s+sqr\s+norm\s+for\s+ref_proj_den_oep\s+Without_R_cutoff_SYMMETRIZED\s+rsqrscaled\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "converged": [r"(?im)^\s*SCF\s+Converged\s"],
    "not_converged": [r"(?im)^\s*SCF\s+NOT\s+Converged\s"],
    "s_ovrlp": [r"(?is)Eigenvalues of S\^I-matrix.*?\n[-\s]*\n\s*1\s+(?:\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "a_matrix": [r"(?is)Eigenvalues of A\^\{III\}-matrix.*?\n[-\s]*\n\s*1\s+(?:\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
}
#u_frozen projected onto grid...metrics
#radial 86 directions for ref_proj_den_oep dnorm     0.1634730334089260
#radial 86 directions for ref_proj_den_oep symmetrized KSINV N_elec     8.0019983274337871
#radial 86 directions for ref_proj_den_oep SYMMETRIZED REF N_elec     7.9999999999966827
#radial 86 directions for ref_proj_den_oep Without_R_cutoff_SYMMETRIZED rscaled     0.5130122328331468
#radial 86 directions for ref_proj_den_oep Without_R_cutoff_SYMMETRIZED rtimes     0.1548696139729197
#radial 86 directions for ref_proj_den_oep Without_R_cutoff_SYMMETRIZED rsqrscaled    11.1922452478673282
#radial 86 directions for ref_proj_den_oep Without_R_cutoff_SYMMETRIZED sqrtrscaled     0.2509755387168288
# radial 86 directions Without_R_cutoff_SYMMETRIZED dnorm   0.0040572562437002
# radial 86 directions symmetrized KSINV N_elec   7.9999999999967191
# radial 86 directions SYMMETRIZED REF N_elec   7.9999999999966827
# radial 86 directions Without_R_cutoff_SYMMETRIZED rscaled   0.0042791267485783
# radial 86 directions Without_R_cutoff_SYMMETRIZED rtimes   0.0079538060314632
# radial 86 directions Without_R_cutoff_SYMMETRIZED rsqrscaled   0.0149157528624285

def _detect_conv(out_text: str, yes_pats: Iterable[str], no_pats: Iterable[str]) -> Optional[bool]:
    for pat in no_pats:
        if re.search(pat, out_text): return False
    for pat in yes_pats:
        if re.search(pat, out_text): return True
    return None

def _first_float(text: str, patterns: Sequence[str]) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try: return float(m.group(1))
            except Exception: pass
    return None

def parse_first_eig_of_S(out_text, patterns):
    for pat in patterns:
        m = re.search(pat,out_text)
    if m:
        try: return float(m.group(1).replace('D', 'E').replace('d', 'e'))
        except Exception: pass
    return None

def parse_opt_first_eig_of_A(out_text, patterns):
    for pat in patterns:
        m = re.findall(pat,out_text)
    if m:
        try: return float(m[-1].group(1).replace('D', 'E').replace('d', 'e'))
        except Exception: pass
    return None


def parse_last_first_eigA_from_lines(out_text: str, patterns: Iterable[str]) -> Optional[float]:
    last_val = None
    for pat in patterns:
        for m in re.finditer(pat, out_text, flags=re.MULTILINE):
            s = m.group(1).replace("D", "E").replace("d", "e")
            try:
                last_val = float(s)
            except ValueError:
                pass
    return last_val


def parse_last_all_eigA(out_text: str) -> Optional[List[float]]:
    """
    Parse ALL eigenvalue blocks of A^{III}-matrix in the Molpro output,
    return eigenvalues from the LAST block (= converged KSINV iteration)
    as a plain list of floats (ascending order).
    Requires VERB,3 or higher in the KSINV input block.
    """
    import numpy as _np
    block_pat = re.compile(
        r"Eigenvalues of A\^\{III\}-matrix\s*\n"
        r"\s*-+\s*\n"
        r"((?:\s*\d+\s+[+-]?\d+\.\d+[EeDd][+-]?\d+\s+[TF]\s*\n)+)",
        re.IGNORECASE,
    )
    line_pat = re.compile(
        r"\s*(\d+)\s+([+-]?\d+\.\d+[EeDd][+-]?\d+)\s+([TF])"
    )
    last_block = None
    for m in block_pat.finditer(out_text):
        last_block = m.group(1)
    if last_block is None:
        return None
    eigs = []
    for m in line_pat.finditer(last_block):
        val = float(m.group(2).replace("D", "E").replace("d", "e"))
        eigs.append(val)
    if not eigs:
        return None
    eigs.sort()
    return eigs


def parse_metrics(out_text: str, phase = "log", parsing_type = "L2") -> Dict[str, Optional[float]]:
    if parsing_type == "L1":
        patterns = DEFAULT_PATTERNS_L1
    elif parsing_type == "L2":
        patterns = DEFAULT_PATTERNS_L2
    else:
        raise ValueError(f"Unknown parsing type: {parsing_type}")
    dvext = _first_float(out_text, patterns["dvext"])
    du    = _first_float(out_text, patterns["du"])
    dlieb = _first_float(out_text, patterns["dlieb"])
    dnorm = _first_float(out_text, patterns["dnorm"])
    rscaled_dnorm = _first_float(out_text, patterns["rscaled_dnorm"])
    sqrtrscaled_dnorm = _first_float(out_text, patterns["sqrtrscaled_dnorm"])
    rtimes_scaled_dnorm = _first_float(out_text, patterns["rtimes_scaled_dnorm"])
    rsqr_scaled_dnorm = _first_float(out_text, patterns["rsqr_scaled_dnorm"])
    ref_proj_dnorm = _first_float(out_text, patterns["ref_proj_dnorm"])
    ref_proj_rscaled_dnorm = _first_float(out_text, patterns["ref_proj_rscaled_dnorm"])
    ref_proj_sqrtrscaled_dnorm = _first_float(out_text, patterns["ref_proj_sqrtrscaled_dnorm"])
    ref_proj_rtimes_scaled_dnorm = _first_float(out_text, patterns["ref_proj_rtimes_scaled_dnorm"])
    ref_proj_rsqr_scaled_dnorm = _first_float(out_text, patterns["ref_proj_rsqr_scaled_dnorm"])
    conv  = _detect_conv(out_text, patterns["converged"], patterns["not_converged"])
    first_eig_of_S = parse_first_eig_of_S(out_text, patterns["s_ovrlp"])
    first_eig_of_A = parse_first_eig_of_S(out_text, patterns["a_matrix"])
    opt_first_eig_of_A = parse_last_first_eigA_from_lines(out_text, patterns["a_matrix"])
    all_eig_of_A = parse_last_all_eigA(out_text)
    logger = logging.getLogger("oep-opt") if phase == "log" else logging.getLogger("oep-opt.grad")
#    if logging:
    if phase == "log":
        logger.info("Parsed metrics: dvext=%s, du=%s, dlieb=%s, dnorm= %s, rscaled_dnorm=%s, sqrtrscaled_dnorm=%s, rtimes_scaled_dnorm=%s, rsqr_scaled_dnorm=%s, ref_proj_dnorm=%s, ref_proj_rscaled_dnorm=%s, ref_proj_sqrtrscaled_dnorm=%s, ref_proj_rtimes_scaled_dnorm=%s, ref_proj_rsqr_scaled_dnorm=%s, converged=%s",
                    dvext, du, dlieb, dnorm, rscaled_dnorm, sqrtrscaled_dnorm, rtimes_scaled_dnorm, rsqr_scaled_dnorm, ref_proj_dnorm, ref_proj_rscaled_dnorm, ref_proj_sqrtrscaled_dnorm, ref_proj_rtimes_scaled_dnorm, ref_proj_rsqr_scaled_dnorm, conv)
        #logger.info("ref_proj Parsed metrics: ref_proj_dnorm=%s, ref_proj_rscaled_dnorm=%s, ref_proj_sqrtrscaled_dnorm=%s, ref_proj_rtimes_scaled_dnorm=%s, ref_proj_rsqr_scaled_dnorm=%s",
        #            ref_proj_dnorm, ref_proj_rscaled_dnorm, ref_proj_sqrtrscaled_dnorm, ref_proj_rtimes_scaled_dnorm, ref_proj_rsqr_scaled_dnorm)   
        logger.info("First eigen value of S is %s",first_eig_of_S)
        logger.info("First eigen value of A is %s",first_eig_of_A)
        logger.info("OPT First eigen value of A is %s",opt_first_eig_of_A)
        if all_eig_of_A is not None:
            logger.info("All A^{III} eigenvalues: %s", ", ".join(f"{v:.6e}" for v in all_eig_of_A))
        else:
            logger.info("All A^{III} eigenvalues: None (VERB>=3 required)")
    #print(f"[DEBUG] Parsed metrics: dvext={dvext}, du={du}, dlieb={dlieb}, dnorm={dnorm}, converged={conv}")
    return {"dvext": dvext, "du": du, "dlieb": dlieb,
             "dnorm": dnorm, "rscaled_dnorm": rscaled_dnorm, "sqrtrscaled_dnorm": sqrtrscaled_dnorm,"rtimes_scaled_dnorm": rtimes_scaled_dnorm,"rsqr_scaled_dnorm": rsqr_scaled_dnorm,
             "ref_proj_dnorm": ref_proj_dnorm, "ref_proj_rscaled_dnorm": ref_proj_rscaled_dnorm, "ref_proj_sqrtrscaled_dnorm": ref_proj_sqrtrscaled_dnorm,
             "ref_proj_rtimes_scaled_dnorm": ref_proj_rtimes_scaled_dnorm, "ref_proj_rsqr_scaled_dnorm": ref_proj_rsqr_scaled_dnorm,
             "converged": conv, "first_eig_of_S": first_eig_of_S, "first_eig_of_A": first_eig_of_A,
             "opt_first_eig_of_A": opt_first_eig_of_A, "all_eig_of_A": all_eig_of_A}