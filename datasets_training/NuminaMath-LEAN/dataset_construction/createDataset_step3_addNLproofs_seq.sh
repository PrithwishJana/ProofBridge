#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-20:00
#SBATCH --job-name=lean_datasetStep3
#SBATCH --output=tmpFolder/lean_datasetStep3_%A.out
#SBATCH --error=tmpFolder/lean_datasetStep3_%A.err

# Load required modules
module load gcc arrow/15.0.1 opencv/4.11.0

# Activate Lean virtual environment
source ~/lean_env/bin/activate

# Loop over sequential indices (0 to 104)
for task_id in $(seq 0 104); do
    # Compute strt_indx for this task (in multiples of 1000)
    strt_indx=$(( task_id * 1000 ))

    # Pad to 6 digits for filenames
    strt_indx_padded=$(printf "%06d" $strt_indx)

    echo "Running task $task_id (strt_indx=$strt_indx_padded)"

    # Run Python script
    python ./datasets_training/NuminaMath-LEAN/createDataset_step3.py \
           --strt_indx $strt_indx 2>&1 \
           | tee ./datasets_training/NuminaMath-LEAN/createDataset_step3LOG_${strt_indx_padded}.txt
done
