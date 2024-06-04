#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=experiments/train_network_no_transf

source activate artsagenet

cd /home/tliberatore2/Artistic_Influence_prediction

#features=("image_features" "text_features" "image_text_features")
features=("image_features")
for feature in "${features[@]}"
do 
    python Triplet_Network.py --dataset_name "wikiart" --feature "$feature" --num_examples 10 --positive_based_on_similarity 
    python evaluation.py --dataset_name "wikiart"  --feature "$feature" --num_examples 10 --positive_based_on_similarity 

done

conda deactivate 