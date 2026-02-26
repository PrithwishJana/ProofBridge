#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-8:00
#SBATCH --job-name=jtemb_train
#SBATCH --output=tmpFolder/jtemb_train_%A.out
#SBATCH --error=tmpFolder/jtemb_train_%A.err

# Load required modules
module load gcc arrow/15.0.1 opencv/4.11.0

# Activate Lean virtual environment
source ~/lean_env/bin/activate

nvidia-smi

python ./joint_embedding/step3_train.py \
    --csv_path ./joint_embedding/step2REPL_Parts \
    --use_wandb \
    --wandb_project "nl-proof-embeddings" \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-5 \
    --nl_trainable_layers 3 \
    --proof_trainable_layers 2