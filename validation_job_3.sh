#!/bin/bash

# Job Flags
#SBATCH --job-name=stem_validation
#SBATCH -p mit_preemptable
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# Set up environment
module load miniforge
source /home/mbosli/DeepMoodPredictor/env/bin/activate

# Run the application
for i in {0..99}
do
    python loo_validation.py $i 'masks/MVP_rois/stem-mask.nii.gz'
done