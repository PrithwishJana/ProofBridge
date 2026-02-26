#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-16:00
#SBATCH --job-name=jtEmb_repl
#SBATCH --output=tmpFolder/jtEmb_repl_%A_%a.out
#SBATCH --error=tmpFolder/jtEmb_repl_%A_%a.err
#SBATCH --array=0-37

# Load required modules
module load gcc arrow/15.0.1 opencv/4.11.0

# Activate Lean virtual environment
source ~/lean_env/bin/activate

# Compute strt_indx for this task (in multiples of 1000)
strt_indx=$(( SLURM_ARRAY_TASK_ID * 1000 ))

# Pad to 6 digits for filenames
strt_indx_padded=$(printf "%06d" $strt_indx)

# Run Python script
python ./joint_embedding/step1REPL_prepareDAGs.py \
       --start_indx $strt_indx 2>&1 \
       | tee ./joint_embedding/step1REPL_Parts/LOG_step1REPL_Part-${strt_indx_padded}.txt
