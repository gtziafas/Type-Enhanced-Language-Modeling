#!/bin/bash
#SBATCH --job-name="8layers_3"
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1  

module purge
module load Python PyTorch
 
python ~/Lassy-Large/Type-Enhanced-Language-Modeling/main.py -s '8layers_3' -l './TypeLM/checkpoints/8layers_2.pth'

