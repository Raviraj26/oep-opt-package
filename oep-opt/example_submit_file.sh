#!/usr/bin/env bash
#SBATCH -J oep_O
#SBATCH -p mem256
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

module purge
# load your molpro env here; then your python venv:
# module load molpro/2024
source ~/venvs/oep/bin/activate

oep-opt optimize   --run-sh /path/to/run.sh   --template molpro_template.inp   --mode free_exponents --K 8   --init-exps "100,33.33,11.11,3.7037,1.2346,0.4115,0.1372,0.0457"
