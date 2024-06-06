#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=experiments/random_artists

source activate artsagenet
cd /home/tliberatore2/Artistic_Influence_prediction

#features=("image_features"  "image_text_features")
#feature_extractors=("stratified_artists" "popular_artists")
features=("image_text_features")
for feature in "${features[@]}"
do 
    #python create_data_loader.py --dataset_name "wikiart" --feature "$feature" --data_split "random_artists" --num_examples 100 --positive_based_on_similarity
    # python create_data_loader.py --dataset_name "wikiart" --feature "$feature" --data_split "popular_artists" --num_examples 100 
    # python create_data_loader.py --dataset_name "wikiart" --feature "$feature" --data_split "popular_artists" --num_examples 10 --positive_based_on_similarity
    # python create_data_loader.py --dataset_name "wikiart" --feature "$feature" --data_split "popular_artists" --num_examples 10

    #python Triplet_Network.py --dataset_name "wikiart" --feature "$feature" --data_split "random_artists" --num_examples 100 --positive_based_on_similarity
    # python Triplet_Network.py --dataset_name "wikiart" --feature "$feature" --data_split "popular_artists" --num_examples 100 
    # python Triplet_Network.py --dataset_name "wikiart" --feature "$feature" --data_split "popular_artists" --num_examples 10 --positive_based_on_similarity
    # python Triplet_Network.py --dataset_name "wikiart" --feature "$feature" --data_split "popular_artists" --num_examples 10

    python evaluation.py --dataset_name "wikiart" --feature "$feature" --data_split "random_artists" --num_examples 100 --positive_based_on_similarity
    #python evaluation.py --dataset_name "wikiart" --feature "$feature" --data_split "unpopular_artists" --num_examples 100 --positive_based_on_similarity
    # python evaluation.py --dataset_name "wikiart" --feature "$feature" --data_split "ResNet34_newsplit" --num_examples 100 
    # python evaluation.py --dataset_name "wikiart" --feature "$feature" --data_split "ResNet34_newsplit" --num_examples 10 --positive_based_on_similarity
    # python evaluation.py --dataset_name "wikiart" --feature "$feature" --data_split "ResNet34_newsplit" --num_examples 10


done






conda deactivate