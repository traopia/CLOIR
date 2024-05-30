# Influence beyond similarity: A Knowledge Based informed model to predict influence between artworks: an Information Retrieval approach
In this work a method to predict influence between artworks has been introduced.
In history of art, experts claim that some artists have been influenced by other artist throughout their artistic career. In this work we explore what does that mean at a computational and operational level. We thus introduce a method that suggests which artworks are related by an influence relation, by exploiting a ground truth that operates at artist level.


# Dataset
1. Wikiart: All images are taken from the [WikiArt Dataset (Refined)](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) github repo. The corresponding metadata has been scraped from [Wikiart] (https://www.wikiart.org). The influence between artist have been retrieved via a scraping from WikiData and WikiArt exploiting the relation [Influenced by](https://www.wikidata.org/wiki/Property:P737)

2. [iDesigner](https://www.kaggle.com/competitions/idesigner/data): 



# PIPELINE
Data extraction and preparation - run the code only once

```
python get_influence_wikidata.py
python dataset_extracted_features.py
```
Steps of experiments are performed with different setups, varying: dataset, feature, train_split, sampling_strategy, num_example_sample

1. Data Loader
```
python create_data_loader.py --dataset_name "wikiart" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 100 --positive_based_on_similarity

```
2. Training model 

```
python Triplet_Network.py --dataset_name "wikiart" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 100 --positive_based_on_similarity

```

3. Evaluation
```
python evaluation.py --dataset_name "wikiart" --feature "$feature" --feature_extractor_name "random_artists" --num_examples 100 --positive_based_on_similarity

```