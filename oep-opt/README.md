# oep-opt

Modular OEP aux-basis s-shell optimizer (SLURM + Molpro).

Install (dev):
```
pip install -e .
```

Run:
```
oep-opt optimize   --run-sh /path/to/run.sh   --template molpro_template.inp   --mode free_exponents --K 8   --init-exps "100,33.33,11.11,3.7037,1.2346,0.4115,0.1372,0.0457"
```
