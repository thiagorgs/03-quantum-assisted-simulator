#!/bin/bash
#SBATCH -J qas_merge_l11_k3
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH -o logs/qas_merge_l11_k3_%j.out
#SBATCH -e logs/qas_merge_l11_k3_%j.err

set -euo pipefail

cd ~/ws_qas
source .venv/bin/activate
mkdir -p logs

python scripts/merge_qas_chunks.py \
  --input-dir data/ClosedBC/chunks \
  --pattern "Mz_Nqu=11_Nrea=1000_K=3_chunk_*.npz" \
  --output data/ClosedBC/Mz_Nqu=11_Nrea=1000_K=3_ClosedBC_QAS_true.npz
