# OEP Basis Optimization

This repo contains scripts and package code for optimizing s-type basis
functions (even-tempered) using Molpro KSINV.

## Layout
- `oep_opt_package/` – reusable code (I/O, objective, parsers).
- `oep_opt_latest.py` – entry script for running one optimization.
- `templates/` – Molpro template files, SLURM launch.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python oep_opt_latest.py --help

