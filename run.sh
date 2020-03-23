#!/bin/bash
#SBATCH --job-name="epoch4"
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1  

module purge
module load Python PyTorch
 
python ~/Lassy-Large/Type-Enhanced-Language-Modeling/main.py -s 'TypeLM_wd1e-7_4' -l './checkpoints/TypeLM_wd1e-7_3.pth'

