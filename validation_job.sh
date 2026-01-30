#!/bin/bash

# Job Flags
#SBATCH --job-name=whole_validation
#SBATCH -p mit_normal_gpu
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# Set up environment
module load miniforge
source /home/mbosli/DeepMoodPredictor/env/bin/activate

# Run the application
for i in {o..99}
do
    python loo_validation.py $i
done