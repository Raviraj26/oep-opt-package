import argparse, hashlib, json, re
from pathlib import Path
from typing import List
import numpy as np

#def stable_tag_from_theta(theta) -> str:
#    tag_bytes = json.dumps(list(map(float, theta))).encode("utf-8")
#    return hashlib.sha1(tag_bytes).hexdigest()[:10]

def stable_tag_from_theta(theta: np.ndarray) -> str:
    t = np.asarray(theta, dtype=np.float64)
    return hashlib.sha1(t.tobytes()).hexdigest()[:10]

def parse_init_exps_str(s: str) -> List[float]:
    toks = [t.strip() for t in re.split(r"[ ,]+", s.strip()) if t.strip()]
    return [float(t) for t in toks]

def read_exps_from_file(path: Path, elem: str) -> List[float]:
    txt = path.read_text()
    for line in txt.splitlines():
        s = line.strip()
        if not s: continue
        if s.lower().startswith('s,'):
            parts = [p.strip().rstrip('.') for p in s.split(',')]
            if len(parts) >= 3 and parts[1].upper() == elem.upper():
                return [float(p) for p in parts[2:] if p]
    # Molpro table fallback
    exps = []
    in_table = False
    for raw in txt.splitlines():
        line = raw.strip("\n")
        if not in_table:
            if ("Centre" in line and "exponent" in line and "contractions" in line):
                in_table = True
            continue
        if not line.strip(): continue
        if set(line.strip()) == {"="}: continue
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].lower() == 's':
            exps.append(float(parts[2]))
    if exps: return exps
    # plain numbers
    for line in txt.splitlines():
        s = line.strip().rstrip('.')
        if not s: continue
        toks = [t for t in s.replace(',', ' ').split() if t]
        nums = []
        ok = True
        for t in toks:
            try: nums.append(float(t))
            except Exception: ok = False; break
        if ok and nums: return nums
    raise ValueError(f"Could not parse exponents from {path}")
