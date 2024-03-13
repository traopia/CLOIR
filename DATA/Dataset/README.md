WikiArt Dataset
====

## Usage

This repository contains the dataset used for the paper "Graph Neural Networks for Knowledge Enhanced Visual Representation of Paintings". All models were evaluated on three subsets, namely WikiArt<sup>Full</sup>, WikiArt<sup>Modern</sup> and WikiArt<sup>Artists</sup>. Each subset is provided as a .csv file (with the prefix of its respective version). A detailed description for each subset is given in the next sections. All images are taken from the [WikiArt Dataset (Refined)](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) github repo.


| Attribute  | WikiArt<sup>Full</sup> | WikiArt<sup>Modern</sup> | WikiArt<sup>Artists</sup> |
|:-|:-|:-|:-|
| Artworks   | 75,921                         | 45,865                           | 17,785                            |
| Artists    | 750                           | 462                             | 23                               |
| Styles     | 20                            | 13                              | 12                               |
| Dates      | 587                           | 150                             | 240                              |
| Timeframes | 13                            | 4                               | 8                                |
| Tags       | 4,879                          | 3,652                            | 2,370                             |



> **Note**: In order to get the most out of this repo, you need first to download the image folder provided in [WikiArt Dataset (Refined)](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset).



## WikiArt<sup>Full</sup>

The WikiArt<sup>Full</sup> dataset can be found in the ```wikiart_full.csv``` file and it contains the following attributes. Examples of the annotations per relevant attribute for the Pablo Picasso’s painting *Les Demoiselles d’Avignon*(1907) as can be found in the WikiArt collection, are given in parentheses.

**Image paths**
* image (containing the names of the 75,921 photo reproductions used, e.g. ```pablo-picasso_the-girls-of-avignon-1907.jpg```).
* relative_path (the relative path for the [WikiArt Dataset (Refined)](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) image folder for each image, e.g. ```Cubism/pablo-picasso_the-girls-of-avignon-1907.jpg```).

**Attributes of Interest**
* style_classification (containing the stylistic movement annotation for each painting, e.g. ```cubism```).
* artist_attribution   (containing the artist attribution annotation for each painting, e.g. ```pablo-picasso```).
* timeframe_estimation (containing the respective timeframe annotation for each painting in half-centuries, e.g. ```1900-1950```).
* tag_prediction (containing the tag annotation(s) for each painting, e.g. ```female-nude```).
* mode (specifying the train, val or test set that each painting was a member of, e.g. ```train```). 

**General Attributes**
* date (the creation year for each painting; if any, e.g. ```1907```).
* artist_name  (the unmasked artist attributed to each painting, e.g. ```pablo-picasso```).
* additional_styles (any additional stylistic movement that a painting was attributed to; if any, e.g. ```None```).
* artist_school  (the school of the artist authored each painting, e.g. ```spanish```).
* tags (all tags associated to each painting; if any and alongside the 54 unique tags that were used for the tag prediction task, e.g. ```female-nude```).

> **Note**: There are 1,111 unique artists in total on the WikiArt<sup>Full</sup> dataset. However, after splitting the data to train/val/test there are only 749 remaining artists with at least an artwork in each dataset split. For artist attribution, we kept these artists and masked out all the other artists by creating a special group entry ```unk```.


## WikiArt<sup>Modern</sup>

The WikiArt<sup>Modern</sup> dataset can be found in the ```wikiart_modern.csv``` file and it contains the following attributes. Examples of the annotations per relevant attribute for the Pablo Picasso’s painting *Les Demoiselles d’Avignon*(1907) as can be found in the WikiArt collection, are given in parentheses.

**Image paths**
* image (containing the names of the 45,865 photo reproductions used, e.g. ```pablo-picasso_the-girls-of-avignon-1907.jpg```).
* relative_path (the relative path for the [WikiArt Dataset (Refined)](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) image folder for each image, e.g. ```Cubism/pablo-picasso_the-girls-of-avignon-1907.jpg```).

**Attributes of Interest**
* style_classification (containing the stylistic movement annotation for each painting, e.g. ```cubism```).
* artist_attribution   (containing the artist attribution annotation for each painting, e.g. ```pablo-picasso```).
* date (the creation year for each painting, e.g. ```1907```).
* mode (specifying the train, val or test set that each painting was a member of, e.g. ```train```).

**General Attributes**
* timeframe (containing the respective timeframe annotation for each painting in half-centuries, e.g. ```1900-1950```).
* artist_name  (the unmasked artist attributed to each painting, e.g. ```pablo-picasso```).
* additional_styles (any additional stylistic movement that a painting was attributed to; if any, e.g. ```None```).
* timeframe (containing the respective timeframe annotation for each painting in half-centuries, e.g. ```1900-1950```).
* artist_school  (the school of the artist authored each painting, e.g. ```spanish```).
* tags (all tags associated to each painting, e.g. ```female-nude```).

> **Note**: There are 45,959 paintings in the WikiArt<sup>Full</sup> dataset with a known creation year annotation between the year 1850 and 1999. However, there are only only four paintings that belong to the following stylistic movements; Mannerism (Late Renaissance), Baroque, Early Renaissance and Northern Renaissance, respectively. We discard these paintings alongside 90 paintings that are attributed to Ukiyo-e stylistic movement (with a total of 1,167 Ukiyo-e paintings in the WikiArt<sup>Full</sup> dataset that were created either before the year 1850 or with not known information about their respective creation year). Finally, as in WikiArt<sup>Full</sup>, we group together artists with no artworks in each train/val/test splits and mask them with a special ```unk``` representation.


## WikiArt<sup>Artists</sup>

The WikiArt<sup>Artists</sup> dataset can be found in the ```wikiart_artist.csv``` file and it contains the following attributes. Examples of the annotations per relevant attribute for the Pablo Picasso’s painting *Les Demoiselles d’Avignon*(1907) as can be found in the WikiArt collection, are given in parentheses.

**Image paths**
* image (containing the names of the 17,785 photo reproductions used, e.g. ```pablo-picasso_the-girls-of-avignon-1907.jpg```).
* relative_path (the relative path for the [WikiArt Dataset (Refined)](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) image folder for each image, e.g. ```Cubism/pablo-picasso_the-girls-of-avignon-1907.jpg```).

**Attributes of Interest**
* style_classification (containing the stylistic movement annotation for each painting, e.g. ```cubism```).
* artist_attribution   (containing the artist attribution annotation for each painting, e.g. ```pablo-picasso```).
* timeframe_estimation (containing the respective timeframe annotation for each painting in half-centuries, e.g. ```1900-1950```).
* mode (specifying the train, val or test set that each painting was a member of, e.g. ```train```).

**General Attributes**
* date (the creation year for each painting, if any; e.g. ```1907```).
* additional_styles (any additional stylistic movement that a painting was attributed to; if any, e.g. ```None```).
* artist_school  (the school of the artist authored each painting, e.g. ```spanish```).
* tags (all tags associated to each painting, e.g. ```female-nude```).

> **Note**: We observed that in the WikiArt collection there are three artists, namely, Henry Matisse (491), Camille Corot (480), and Alfred Sisley (465), that are attributed to more artworks compared to Salvador Dalí (463). However, in order to be consistent with previous work[1,2] on artist attribution task for the 23 most productive artists in the WikiArt collection, we did not consider their works, but directly used Salvador Dalí as the 23rd most productive artist. The number of artworks attributed to the afforementioned artists is given in parentheses for each artist.



## References

[1] [Wei Ren Tan, Chee Seng Chan, Hernán E Aguirre, and Kiyoshi Tanaka. 2016. Ceci n’est pas une pipe: A deep convolutional network for fine-art paintings classification. In 2016 IEEE international conference on image processing (ICIP). IEEE, 3703–3707.](https://github.com/cs-chan/ArtGAN/tree/master/ICIP-16)

[2] [Eva Cetinic, Tomislav Lipic, and Sonja Grgic. 2019. A deep learning perspective on beauty, sentiment, and remembrance of art. IEEE Access 7 (2019), 73694–73710.](https://ieeexplore.ieee.org/document/8731853)
