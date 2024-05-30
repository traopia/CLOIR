from matplotlib import pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

def split_by_strata_artist(df, train_size=0.7):
    df['mode'] = None
    for artist_name, group in df.groupby('artist_name'):
        train_indices, val_indices = train_test_split(group.index, train_size=train_size, random_state=42)    
        df.loc[train_indices, 'mode'] = 'train'
        df.loc[val_indices, 'mode'] = 'val'

    
    return df



def split_by_strata_artist_designer(df, train_size=0.7):
    features = df['image_features'].to_numpy()
    features = np.stack(features)
    features = torch.tensor(features)
    similarity_matrix = cosine_similarity(features)
    n = len(similarity_matrix)
    similarity_matrix = similarity_matrix - np.identity(n)
    df['mode'] = None
    
    for artist_name, group in df.groupby('artist_name'):
        train_indices, val_indices = train_test_split(group.index, train_size=train_size, random_state=42)

        
        df.loc[train_indices, 'mode'] = 'train'
        df.loc[val_indices, 'mode'] = 'val'
    
    id_map = df.index.to_list()
    all_similar_images = []
    for idx, sample in enumerate(df.iterrows()):
        #for each image, find the most similar images above a certain threshold
        threshold = 0.95
        similar_images = np.where(similarity_matrix[idx] > threshold)[0]
        similar_images = [id_map[x] for x in similar_images]
        if len(similar_images) > 1:
            all_similar_images.append(similar_images)
            if df.loc[similar_images[0], 'mode'] == 'train':
                #all similar images are in the same split
                for i in similar_images:
                    df.loc[i, 'mode'] = 'train'
            else:
                df.loc[similar_images[1], 'mode'] = 'val'
                for i in similar_images:
                    df.loc[i, 'mode'] = 'val'
    
    return df



def split_by_artist_random(df, train_size=0.7,random_state=42):
    unique_artists = df['artist_name'].unique()
    train_artists, val_artists = train_test_split(unique_artists, train_size=train_size, random_state=random_state)
    df['mode'] = None
    df.loc[df['artist_name'].isin(train_artists), 'mode'] = 'train'
    df.loc[df['artist_name'].isin(val_artists), 'mode'] = 'val'
    return df


def split_by_artist_given(df, artist_name):
    df['mode'] = None
    df.loc[df['artist_name'] != artist_name, 'mode'] = 'train'
    df.loc[df['artist_name'] == artist_name, 'mode'] = 'val'
    
    return df   


def plot_examples(dataset_name, query, positive_indexes, df):
    # Plot single image
    if dataset_name == 'wikiart':
        general_image_path = '/home/tliberatore2/Reproduction-of-ArtSAGENet/wikiart/'
    elif dataset_name == 'fashion':
        general_image_path = 'DATA/Dataset/iDesigner/designer_image_train_v2_cropped/'
    plt.figure(figsize=(10, 5))
    plt.imshow(Image.open(general_image_path + df.loc[query].relative_path))
    plt.axis('off')
    plt.title(str(df.loc[query].artist_name + ', influencers: ' + str(df.loc[query].influenced_by)))
    plt.show()

    # Plot grid of images
    fig, axes = plt.subplots(3,3, figsize=(20, 20))  # 5 rows, 2 columns
    for i, ax in enumerate(axes.flatten()):
        if i < len(positive_indexes):
            image_path = general_image_path + df.iloc[positive_indexes[i]].relative_path
            image = Image.open(image_path)
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(str(i+1)+" "+ df.iloc[positive_indexes[i]].artist_name)
        else:
            ax.axis('off')  # Hide unused subplots

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()
