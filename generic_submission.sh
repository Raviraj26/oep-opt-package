#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1000:00:00
#SBATCH --partition=mem384
#SBATCH --exclusive
#SBATCH --job-name=oepopt_%x

set -euo pipefail

: "${ELEM:?Set ELEM (e.g. ELEM=O)}"
CASE_ENV="cases/${ELEM}.env"
[[ -f "$CASE_ENV" ]] || { echo "Missing $CASE_ENV"; exit 2; }
source "$CASE_ENV"


# --- derive K from INIT_EXPS (space or comma separated) ---
: "${INIT_EXPS:?Set INIT_EXPS in cases/${ELEM}.env (e.g. '1e-3 2e-3 ...' or '1e-3,2e-3,...')}"

# normalize commas -> spaces, squeeze whitespace, split into array
INIT_EXPS_NORM="$(echo "$INIT_EXPS" | tr ',' ' ' | xargs)"
read -r -a EXPS_ARR <<< "$INIT_EXPS_NORM"
K="${#EXPS_ARR[@]}"

if (( K <= 0 )); then
  echo "[ERR] Could not parse any exponents from INIT_EXPS='$INIT_EXPS'"
  exit 2
fi

# --- choose workers based on FD scheme ---
PARALLEL_EVAL=true
PAR_METHOD="central"     # set this once and reuse (central|forward)
#PAR_METHOD="forward"

if [[ "$PAR_METHOD" == "central" ]]; then
  WORKERS=$((2 * K))
elif [[ "$PAR_METHOD" == "forward" ]]; then
  WORKERS=$((K))
else
  echo "[ERR] Unknown PAR_METHOD='$PAR_METHOD' (expected central|forward)"
  exit 2
fi

export OMP_STACKSIZE=3000M
ulimit -s unlimited
cd "$SLURM_SUBMIT_DIR"
echo "$SLURM_JOB_NODELIST" > "$SLURM_SUBMIT_DIR/machines_jobid"

export OEP_MOLPRO_MEM_MB=10000
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OEP_CPUS_PER_EVAL=1
export OEP_SCRATCH_ROOT="/scratch/${USER}/${SLURM_JOB_ID}"

# shared knobs (same for all elements)
TEMPLATE=molpro_template.inp
RUN_SH=/home/mandalia/WORK_DIR/run_new.sh
SBATCH_CMD="bash"

# weights/optimizer
W_DVEXT=00.0
W_DU=0.0
W_LIEB=0.0
W_NORM=65.0
W_RSCALED=16.0
W_SQRT=0.0
W_RTIMES=1.0
W_RSQR=1.0
W_REF_PROJ_NORM=100.0
W_REF_PROJ_RSCALED=5.0
W_REF_PROJ_SQRT=0.0
W_REF_PROJ_RTIMES=1.0
W_REF_PROJ_RSQR=1.0
MAXITER="${MAXITER:-300}"
EPS="${EPS:-1e-4}"
POLL_S=20
MAX_WAIT_S=0
METHOD="BFGS"
#METHOD="L-BFGS-B"
GTOL=1e-6
#Tunable oriderable variable 
R_DNORMCUTOFF="${R_DNORMCUTOFF:-20.0}"

# penalties / parallel
RED_A=1.4
RED_KNOB=False
RED_B=0.3
RED_C=1e-1
A_TYP=quartic
A_EXP=1.0
A_COEFF=1e+1
A_KNOB=False
S_EXP=9
S_COEFF=5e-0
S_KNOB=True
PARSING_TYP=L2

#WORKDIR="mp2fit_${ELEM}_${ORBITAL_PARENT}_${SLURM_JOB_ID}_w_radial_grid_w_43_25_25_7_s_8"
#WORKDIR="mp2fit_${ELEM}_${ORBITAL_PARENT}_${SLURM_JOB_ID}_w_radial_grid_w_43_25_25_7_a_2_c_5minus1_r_a_1point4_c_minus1"

#WORKDIR="Both_oep_and_ref_proj_normalised_new_s_form_7_10percent_w_shift_to_rscaled_1_percent_shift_to_rsqr"
#ORKDIR="norm_weights_s_5_more_weight_oep_run9_a_coupl_expopoint3"
#WORKDIR="norm_weights_s_5_more_weight_oep_run3_add_250_rerun"
#WORKDIR="parsing_type_trial"
#WORKDIR="log_all_grad_eps_test_${ELEM}_${ORBITAL_PARENT}_${SLURM_JOB_ID}_w_43_25_25_7_no_acoupl_s_7_${EPS}_CD_rdnormcutoff_6_mp2_aug"
#WORKDIR="from_1e-6_optimized_${ELEM}_${ORBITAL_PARENT}_${SLURM_JOB_ID}_w_dvext_only_${EPS}_CD_rdnormcutoff_6_mp2_aug"
#WORKDIR="debug_log_all_grad_eps_test_${ELEM}_${ORBITAL_PARENT}_756733_w_dnorm_only_${EPS}_CD_rdnormcutoff_6_mp2_aug"

#WORKDIR="from_optimized_New_radial_grid_${ELEM}_${ORBITAL_PARENT}_${SLURM_JOB_ID}_w_dvext_only_${EPS}_CD_mp2_aug"
#WORKDIR="New_radial_grid_log_all_grad_eps_test_${ELEM}_${ORBITAL_PARENT}_${SLURM_JOB_ID}_w_dvext_only_${EPS}_CD_mp2_aug"

#WORKDIR="debug____opt_${ELEM}_${ORBITAL_PARENT}_${SLURM_JOB_ID}_w_43_25_25_7_no_acoupl_s_7_epsminus4_trial3"
#WORKDIR="L2_normalised_norm_test_${ELEM}_s_9_quartic" 
WORKDIR="2ndrow_S_test_add_steep_n_diffuse"

oep-opt \
  --mode free_exponents \
  --elem "$ELEM" \
  --charge "$CHARGE" \
  --spin "$SPIN" \
  --alpha-occ "$ALPHA_OCC" \
  --beta-occ "$BETA_OCC" \
  --r-dnorm-cutoff "$R_DNORMCUTOFF" \
  --orbital-parent "$ORBITAL_PARENT" \
  --dm-file "$DM_FILE" \
  --input-case-file "$CASE_ENV" \
  --e-ref "$E_REF" \
  --K "$K" \
  --template "$TEMPLATE" \
  --workdir "$WORKDIR" \
  --w-dvext "$W_DVEXT" --w-du "$W_DU" --w-lieb "$W_LIEB" --w-norm "$W_NORM" \
  --w-rscaled-norm "$W_RSCALED" --w-sqrtrscaled-norm "$W_SQRT" \
  --w-rtimes-scaled-norm "$W_RTIMES" --w-rsqr-scaled-norm "$W_RSQR" \
  --w-ref-proj-norm "$W_REF_PROJ_NORM" \
  --w-ref-proj-rscaled-norm "$W_REF_PROJ_RSCALED" \
  --w-ref-proj-rsqr-scaled-norm "$W_REF_PROJ_RSQR" \
  --w-ref-proj-sqrtrscaled-norm "$W_REF_PROJ_SQRT" \
  --w-ref-proj-rtimes-scaled-norm "$W_REF_PROJ_RTIMES" \
  --maxiter "$MAXITER" --eps "$EPS" \
  --run-sh "$RUN_SH" \
  --gtol "$GTOL" \
  --sbatch-cmd "$SBATCH_CMD" \
  --init-exps "$INIT_EXPS" \
  --parsing_type "$PARSING_TYP" \
  --poll-s "$POLL_S" --max-wait-s "$MAX_WAIT_S" --method "$METHOD" \
  --redundancy_penalty_coeff_a "$RED_A" --redundancy_penalty_knob "$RED_KNOB" \
  --redundancy_penalty_coeff_b "$RED_B" --redundancy_penalty_coeff_c "$RED_C" \
  --a_coupling_penalty_expo "$A_EXP" --a_coupling_penalty_coeff "$A_COEFF" --knob_for_a_coupling_penalty "$A_KNOB" \
  --a_coupling_penalty_type "$A_TYP" \
  --parallel_eval $PARALLEL_EVAL --parallel_eval_workers $WORKERS --parallel_eval_method "$PAR_METHOD" \
  --s_ovrlp_penalty_expo "$S_EXP" --s_ovrlp_penalty_coeff "$S_COEFF" --knob_for_s_ovrlp_penalty "$S_KNOB"

