#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_arcface.sh — SLURM batch script for ArcFace fine-tuning
#
# Submit:
#   sbatch training/run_arcface.sh                      # 4 GPUs (default)
#   sbatch --gres=gpu:1 training/run_arcface.sh         # single GPU
#   sbatch training/run_arcface.sh --config my.yaml     # custom config
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=arcface_finetune
#SBATCH --output=/ceph/project/P8_DCASE/logs/arcface_%j.out
#SBATCH --error=/ceph/project/P8_DCASE/logs/arcface_%j.err
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --array=1-2%1
#nnnSBATCH --exclude=ailab-l4-09
#SBATCH --dependency=817925_2
set -euo pipefail

echo "==== JOB START ===="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "GPUs    : ${SLURM_JOB_GPUS:-auto}"

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGE="/ceph/container/pytorch/pytorch_25.08.sif"
VENV="/ceph/project/P8_DCASE/p8_env"
# SLURM_SUBMIT_DIR is the directory sbatch was called from (e.g. AAU_P8/).
# BASH_SOURCE[0] resolves to the spool copy and cannot be used under SLURM.
TRAIN_DIR="${SLURM_SUBMIT_DIR}/training"

# ── Config and extra args passed through from sbatch CLI ─────────────────────
CONFIG="${1:---config arcface_config_4l.yaml}"
#CONFIG="${1:---config sslam_config.yaml}"




# If first arg looks like a flag (--) pass everything through; else treat as
# positional config path for convenience.
if [[ "$CONFIG" != --* ]]; then
    CONFIG="--config $CONFIG"
fi
EXTRA_ARGS="${*:2}"

# ── Count allocated GPUs ──────────────────────────────────────────────────────
NGPU=$(echo "${SLURM_JOB_GPUS:-0,1,2,3}" | tr ',' '\n' | wc -l)
echo "Launching with $NGPU GPU(s)"
echo "Config  : $CONFIG"
echo "$(date)"

# ── Launch inside Singularity ─────────────────────────────────────────────────
singularity exec --nv \
    -B /ceph/project/P8_DCASE \
    -B "$VENV:/scratch/p8_env" \
    "$IMAGE" \
    bash -c "
        set -euo pipefail
        cd $TRAIN_DIR
        if [[ $NGPU -gt 1 ]]; then
            /scratch/p8_env/bin/torchrun \
                --standalone \
                --nproc_per_node=$NGPU \
                train_arcface.py $CONFIG $EXTRA_ARGS
        else
            /scratch/p8_env/bin/python -u train_arcface.py $CONFIG $EXTRA_ARGS
        fi
    "

echo "==== DONE — $(date) ===="
