#!/bin/bash
#
#SBATCH --job-name=train-itag-smodel
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=62GB
#SBATCH --output=logs/train-itag-smodel_%a_%A.out
#SBATCH --error=logs/train-itag-smodel_%a_%A.err

module purge
module load r/intel/3.3.2
 
cd $SCRATCH/video/repo/hpc

echo "$SLURM_ARRAY_TASK_ID"

R --no-save -q -f train-itag-smodel.R
