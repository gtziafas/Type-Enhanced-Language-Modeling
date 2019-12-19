#!/bin/bash
#SBATCH --job-name="bubbleshort"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1  

module purge
module load Python PyTorch
 
python ~/Lassy-Large/Type-Enhanced-Language-Modeling/main.py
