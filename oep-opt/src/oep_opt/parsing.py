import re
from typing import Dict, Iterable, Optional, Sequence
import logging
logger = logging.getLogger("oep-opt")

DEFAULT_PATTERNS: Dict[str, list[str]] = {
    "dvext": [r"(?m)^\s*KSINV\s+External\s+energy\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "du":    [r"(?m)^\s*KSINV\s+Hartree\s+energy\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "dlieb": [r"(?m)^\s*KSINV\s+Lieb\s+error\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"],
    "dnorm": [r"(?im)^\s*SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "rscaled_dnorm": [r"(?im)^\s*R_SCALED_SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "sqrtrscaled_dnorm": [r"(?im)^\s*SQRTR_SCALED_SYMMETRIZED\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"],
    "converged": [r"(?im)^\s*SCF\s+Converged\s"],
    "not_converged": [r"(?im)^\s*SCF\s+NOT\s+Converged\s"],
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

def parse_metrics(out_text: str) -> Dict[str, Optional[float]]:
    dvext = _first_float(out_text, DEFAULT_PATTERNS["dvext"])
    du    = _first_float(out_text, DEFAULT_PATTERNS["du"])
    dlieb = _first_float(out_text, DEFAULT_PATTERNS["dlieb"])
    dnorm = _first_float(out_text, DEFAULT_PATTERNS["dnorm"])
    rscaled_dnorm = _first_float(out_text, DEFAULT_PATTERNS["rscaled_dnorm"])
    sqrtrscaled_dnorm = _first_float(out_text, DEFAULT_PATTERNS["sqrtrscaled_dnorm"])
    conv  = _detect_conv(out_text, DEFAULT_PATTERNS["converged"], DEFAULT_PATTERNS["not_converged"])
    logger.info("Parsed metrics: dvext=%s, du=%s, dlieb=%s, dnorm= %s, rscaled_dnorm=%s, sqrtrscaled_dnorm=%s, converged=%s", dvext, du, dlieb, dnorm, rscaled_dnorm, sqrtrscaled_dnorm, conv)
    #print(f"[DEBUG] Parsed metrics: dvext={dvext}, du={du}, dlieb={dlieb}, dnorm={dnorm}, converged={conv}")
    return {"dvext": dvext, "du": du, "dlieb": dlieb, "dnorm": dnorm, "rscaled_dnorm": rscaled_dnorm, "sqrtrscaled_dnorm": sqrtrscaled_dnorm, "converged": conv}
