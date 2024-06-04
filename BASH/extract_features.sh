#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --output=experiments/extract_features_idesigner

source activate artsagenet

cd /home/tliberatore2/Artistic_Influence_prediction

python dataset_extracted_features.py

conda deactivate 