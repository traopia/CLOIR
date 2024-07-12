# CLOIR A Contrastive Learning approach to Object Influence Retrieval
This is the source code for the paper "Influence beyond similarity: a Contrastive Learning approach to Object Influence Retrieval".
We introduce an approach to suggest the existence of influence relations between objects, having access to information about influence between their agents. 
The main steps of the approach include (i) sourcing of influence relations between agents (ii) feature extraction to represent the objects (iii) training of a contrastive network with triplet loss (iv) retrieval of suggested influential objects and evaluation. An ![overview](https://github.com/traopia/CLOIR/blob/main/images/method.pdf) of the approach is provided.


# Datasets
1. Wikiart: All images are taken from the [WikiArt Dataset (Refined)](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) github repo. The corresponding metadata has been scraped from [Wikiart](https://www.wikiart.org). The influence between artist have been retrieved via a scraping from WikiData and WikiArt exploiting the relation [Influenced by](https://www.wikidata.org/wiki/Property:P737). 

2. [iDesigner](https://www.kaggle.com/competitions/idesigner/data): We use the images contained in DATA/Dataset/iDesigner/designer_image_train_v2_cropped.

3. The retrieved influences between agents can be found in DATA/influence_dicts. The code to retrieve the agent influences from Wikidata can be found in get_influence_wikidata.py


 # CLOIR
 Data extraction and preparation - run the code only once

```
python get_influence_wikidata.py
python dataset_extracted_features.py
```
Steps of experiments are performed with different setups, varying: dataset, feature, train_split, sampling_strategy, num_example_sample

1. Data Loader
```
python create_data_loader.py --dataset_name "wikiart" --feature "$feature" --data_split "stratified_artists" --num_examples 100 --positive_based_on_similarity

```
2. Training model 

```
python Triplet_Network.py --dataset_name "wikiart" --feature "$feature" --data_split "stratified_artists" --num_examples 100 --positive_based_on_similarity

```

3. Evaluation
```
python evaluation.py --dataset_name "wikiart" --feature "$feature" --data_split "stratified_artists" --num_examples 100 --positive_based_on_similarity

```










