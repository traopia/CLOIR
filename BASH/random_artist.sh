#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --output=experiments/random_artist

source activate artsagenet
cd /home/tliberatore2/Artistic_Influence_prediction

# python create_data_loader.py --dataset_name "wikiart"  --feature "image_text_features" --feature_extractor_name "random_artists" --num_examples 100 --positive_based_on_similarity
# python Triplet_Network.py --dataset_name "wikiart"  --feature "image_text_features" --feature_extractor_name "random_artists" --num_examples 100 --positive_based_on_similarity
python evaluation.py --dataset_name "wikiart"  --feature "image_text_features" --feature_extractor_name "random_artists" --num_examples 100 --positive_based_on_similarity

conda deactivate