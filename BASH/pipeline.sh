#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=experiments/fashion_evals

source activate artsagenet
cd /home/tliberatore2/Artistic_Influence_prediction

#features=("image_features"  "image_text_features")
feature_extractors=("ResNet34_newsplit" "random_artists")
features=("image_features")
for feature in "${features[@]}"
do 
    # python create_data_loader.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 100 --positive_based_on_similarity
    # python create_data_loader.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 100 
    # python create_data_loader.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 10 --positive_based_on_similarity
    # python create_data_loader.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 10

    # python Triplet_Network.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 100 --positive_based_on_similarity
    # python Triplet_Network.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 100 
    # python Triplet_Network.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 10 --positive_based_on_similarity
    # python Triplet_Network.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 10

    python evaluation.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "ResNet34_newsplit" --num_examples 100 --positive_based_on_similarity
    python evaluation.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "ResNet34_newsplit" --num_examples 100 
    python evaluation.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "ResNet34_newsplit" --num_examples 10 --positive_based_on_similarity
    python evaluation.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "ResNet34_newsplit" --num_examples 10

    python evaluation.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 100 --positive_based_on_similarity
    python evaluation.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 100 
    python evaluation.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 10 --positive_based_on_similarity
    python evaluation.py --dataset_name "fashion" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 10

done






conda deactivate