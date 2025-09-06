# Placeholder for config.py. Full code was in assistant's answer.
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Weights:
    w_dvext: float = 3.0
    w_du: float = 1.0
    w_lieb: float = 1.0
    w_norm: float = 1.0
    w_scaled_norm: float = 1.0
    penalty_noconv: float = 1e4

@dataclass
class JobConfig:
    # chemical/system
    elem: str = "O"
    charge: int = 0
    spin: int = 2
    orbital_parent: str = "aug-cc-pwCV5Z"
    aux_parent: str = "aug-cc-pVDZ/mp2fit"
    dm_file: str = "dm.dat"
    e_ref: float = -75.056798837342

    # paths
    template_text: str = ""
    workroot: Path = Path("runs_O")
    run_sh_path: Path = Path("example_submit_file.sh")

    # optimization
    mode: str = "free_exponents"  # or "even_tempered"
    K: int = 8
    weights: Weights = field(default_factory=Weights)
    sbatch_cmd: str = "sbatch"
    poll_s: int = 20
    max_wait_s: int = 0
    exp_min: float = 1e-6
    exp_max: float = 1e6
    order_penalty: float = 0.0

