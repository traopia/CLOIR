# Influence beyond similarity: How influence between agents can help to suggest influence between objects
We introduce an approach to suggest the existence of influence relations between objects, having access to information about influence between their agents. 
The main steps of the approach include (i) sourcing of influence relations between agents (ii) feature extraction to represent the objects (iii) training of a contrastive network with triplet loss (iv) retrieval of suggested influential objects and evaluation. An ![overview](images/method.pdf) of the approach is provided.


We test the approach with two datasets: Wikiart and iDesigner. The corresponding influence graphs used for the experiments are attached in the repo. 


# Dataset
1. Wikiart: All images are taken from the [WikiArt Dataset (Refined)](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) github repo. The corresponding metadata has been scraped from [Wikiart] (https://www.wikiart.org). The influence between artist have been retrieved via a scraping from WikiData and WikiArt exploiting the relation [Influenced by](https://www.wikidata.org/wiki/Property:P737). 

2. [iDesigner](https://www.kaggle.com/competitions/idesigner/data): We use the images contained in DATA/Dataset/iDesigner/designer_image_train_v2_cropped. 



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
