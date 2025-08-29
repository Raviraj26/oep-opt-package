#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=testjob
#SBATCH --time=1000:00:00
#SBATCH --partition=mem256

export I_MPI_DEBUG=5
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so.0


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_THREAD_LIMIT=$SLURM_CPUS_PER_TASK
export OMP_STACKSIZE=3000M
ulimit -s unlimited
export MV2_CPU_MAPPING=0-$(( ${num} - 1))

cd $SLURM_SUBMIT_DIR
echo $SLURM_JOB_NODELIST > $SLURM_SUBMIT_DIR/machines_jobid

TMPDIR=/scratch/${USER}/${SLURM_JOBID}


python oep_opt_latest.py \
   --mode free_exponents \
   --elem O \
   --charge 0 \
   --spin 2 \
   --orbital-parent aug-cc-pwCV5Z \
   --aux-parent aug-cc-pVDZ/mp2fit \
   --dm-file /home/mandalia/WORK_DIR/AQCC_ROHF_aug-cc-pwCV5Z/o/dm.dat \
   --e-ref -75.056798837342 \
   --K 10 \
   --template molpro_template.inp \
   --workdir DEBUG \
   --w-dvext 1.00 --w-du 0.25 --w-lieb 0.0 --w-norm 1.0 \
   --maxiter 80 \
   --run-sh /home/mandalia/WORK_DIR/Auto_optimize/run.sh \
   --init-exps "1200.000,327.000,109.461,25.578,9.55148,2.9396,1.39638,0.905061,0.421376,0.121792" \
   --poll-s 20 --max-wait-s 0 --method "BFGS" 
	#--init-alpha-hi 1000.00 \
   #--init-exps-file   current_s_basis_O.txt \
   #--init-beta 2. \
   

scp $TMPDIR/* tcsv020:$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR




