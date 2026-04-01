#!/bin/bash
#SBATCH -J qas_l11_k3
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH -o logs/qas_l11_k3_%A_%a.out
#SBATCH -e logs/qas_l11_k3_%A_%a.err

set -euo pipefail

cd ~/ws_qas
source .venv/bin/activate
mkdir -p logs data/ClosedBC/chunks

L=11
K=3
NREA_TOTAL=1000
NCHUNKS=20
CHUNK_SIZE=$((NREA_TOTAL / NCHUNKS))
START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

python scripts/qas_true_disordered_closedbc.py \
  --mode produce \
  --L ${L} \
  --K ${K} \
  --nrea ${NREA_TOTAL} \
  --rea-start ${START} \
  --rea-end ${END} \
  --J 1.0 \
  --h 0.6 \
  --J2 0.3 \
  --tfinal 400 \
  --seed 1 \
  --output data/ClosedBC/chunks/Mz_Nqu=${L}_Nrea=${NREA_TOTAL}_K=${K}_chunk_${START}_${END}.npz
