import re
from typing import Dict, Iterable, Optional, Sequence, List
import logging
#logger = logging.getLogger("oep-opt")


DEFAULT_PATTERNS: Dict[str, list[str]] = {
    "dvext": [r"(?m)^\s*KSINV\s+External\s+energy\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "du":    [r"(?m)^\s*KSINV\s+Hartree\s+energy\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "dlieb": [r"(?m)^\s*KSINV\s+Lieb\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "dnorm": [r"(?im)^\s*SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "rscaled_dnorm": [r"(?im)^\s*R_SCALED_SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "sqrtrscaled_dnorm": [r"(?im)^\s*SQRTR_SCALED_SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "rtimes_scaled_dnorm": [r"(?im)^\s*Rtimes_SCALED_SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "rsqr_scaled_dnorm": [r"(?im)^\s*RSQR_SCALED_SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "converged": [r"(?im)^\s*SCF\s+Converged\s"],
    "not_converged": [r"(?im)^\s*SCF\s+NOT\s+Converged\s"],
    "s_ovrlp": [r"(?is)Eigenvalues of S\^I-matrix.*?\n[-\s]*\n\s*1\s+(?:\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "a_matrix": [r"(?is)Eigenvalues of A\^\{III\}-matrix.*?\n[-\s]*\n\s*1\s+(?:\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
}

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


def parse_metrics(out_text: str, phase = "log") -> Dict[str, Optional[float]]:
    dvext = _first_float(out_text, DEFAULT_PATTERNS["dvext"])
    du    = _first_float(out_text, DEFAULT_PATTERNS["du"])
    dlieb = _first_float(out_text, DEFAULT_PATTERNS["dlieb"])
    dnorm = _first_float(out_text, DEFAULT_PATTERNS["dnorm"])
    rscaled_dnorm = _first_float(out_text, DEFAULT_PATTERNS["rscaled_dnorm"])
    sqrtrscaled_dnorm = _first_float(out_text, DEFAULT_PATTERNS["sqrtrscaled_dnorm"])
    rtimes_scaled_dnorm = _first_float(out_text, DEFAULT_PATTERNS["rtimes_scaled_dnorm"])
    rsqr_scaled_dnorm = _first_float(out_text, DEFAULT_PATTERNS["rsqr_scaled_dnorm"])
    conv  = _detect_conv(out_text, DEFAULT_PATTERNS["converged"], DEFAULT_PATTERNS["not_converged"])
    first_eig_of_S = parse_first_eig_of_S(out_text, DEFAULT_PATTERNS["s_ovrlp"])
    first_eig_of_A = parse_first_eig_of_S(out_text, DEFAULT_PATTERNS["a_matrix"])
    opt_first_eig_of_A = parse_last_first_eigA_from_lines(out_text, DEFAULT_PATTERNS["a_matrix"])
    logger = logging.getLogger("oep-opt") if phase == "log" else logging.getLogger("oep-opt.grad")
#    if logging:
    if phase == "log":
        logger.info("Parsed metrics: dvext=%s, du=%s, dlieb=%s, dnorm= %s, rscaled_dnorm=%s, sqrtrscaled_dnorm=%s, rtimes_scaled_dnorm=%s, rsqr_scaled_dnorm=%s, converged=%s",
                    dvext, du, dlieb, dnorm, rscaled_dnorm, sqrtrscaled_dnorm, rtimes_scaled_dnorm, rsqr_scaled_dnorm, conv)
        logger.info("First eigen value of S is %s",first_eig_of_S)
        logger.info("First eigen value of A is %s",first_eig_of_A)
        logger.info("OPT First eigen value of A is %s",opt_first_eig_of_A)
    #print(f"[DEBUG] Parsed metrics: dvext={dvext}, du={du}, dlieb={dlieb}, dnorm={dnorm}, converged={conv}")
    return {"dvext": dvext, "du": du, "dlieb": dlieb, "dnorm": dnorm, "rscaled_dnorm": rscaled_dnorm, "sqrtrscaled_dnorm": sqrtrscaled_dnorm,"rtimes_scaled_dnorm": rtimes_scaled_dnorm,
            "rsqr_scaled_dnorm": rsqr_scaled_dnorm, "converged": conv, "first_eig_of_S": first_eig_of_S, "first_eig_of_A": first_eig_of_A, "opt_first_eig_of_A": opt_first_eig_of_A}
