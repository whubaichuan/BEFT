#!/bin/env bash

#SBATCH -A naiss2024-22-1182         # find your project with the "projinfo" command
#SBATCH -p alvis              # what partition to use (usually not necessary)
#SBATCH -t 00-03:00:00         # how long time it will take to run
#SBATCH --gpus-per-node=T4:1  # choosing no. GPUs and their type
#SBATCH -J bitfit      # the jobname (not necessary)

module purge
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0 
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 accelerate/0.33.0-foss-2023a-CUDA-12.1.1 
source /mimer/NOBACKUP/groups/naiss2024-22-1182/baichuan/environment/bitfit/bin/activate

python run_glue.py