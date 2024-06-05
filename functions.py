from matplotlib import pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr

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

def split_based_on_popularity(df):
    dict_influenced_by = df.groupby('artist_name')['influenced_by'].first().to_dict()
    dict_influencers = {}
    for key, value in dict_influenced_by.items():
        for v in value:    
            if v not in dict_influencers:
                dict_influencers[v] = []      
            dict_influencers[v].append(key)
    average_influencers = int(np.mean([len(lst) for lst in dict_influencers.values()]))
    filtered_dict = {key: value for key, value in dict_influencers.items() if len(value) > average_influencers}
    train_artists = filtered_dict.keys()
    df['mode'] = None
    df.loc[df['artist_name'].isin(train_artists), 'mode'] = 'train'
    df.loc[~df['artist_name'].isin(train_artists), 'mode'] = 'val'
    return df

def split_based_on_unpopularity(df):
    dict_influenced_by = df.groupby('artist_name')['influenced_by'].first().to_dict()
    dict_influencers = {}
    for key, value in dict_influenced_by.items():
        for v in value:    
            if v not in dict_influencers:
                dict_influencers[v] = []      
            dict_influencers[v].append(key)
    average_influencers = int(np.mean([len(lst) for lst in dict_influencers.values()]))
    filtered_dict = {key: value for key, value in dict_influencers.items() if len(value) > average_influencers}
    train_artists = filtered_dict.keys()
    df['mode'] = None
    df.loc[~df['artist_name'].isin(train_artists), 'mode'] = 'train'
    df.loc[df['artist_name'].isin(train_artists), 'mode'] = 'val'
    return df

def plot_examples(dataset_name, query, positive_indexes, df):
    # Plot single image
    if dataset_name == 'wikiart':
        general_image_path = '/home/tliberatore2/Reproduction-of-ArtSAGENet/wikiart/'
    if dataset_name == "fashion":
        general_image_path = 'DATA/Dataset/iDesigner/designer_image_train_v2_cropped/'

    plt.figure(figsize=(10, 5))
    plt.imshow(Image.open(general_image_path + df.loc[query].relative_path))
    plt.axis('off')
    if dataset_name == 'wikiart':
        title_str = (r"$\bf{Agent:}$ " + df.loc[query].artist_name + '\n'
             r"$\bf{Influencers:}$ " + ',\n'.join(df.loc[query].influenced_by) + 
             '\n' + r"$\bf{date:}$ " + str(df.loc[query].date))
    else:
        title_str = (r"$\bf{Agent:}$ " + df.loc[query].artist_name + '\n'
             r"$\bf{Influencers:}$ " + ',\n'.join(df.loc[query].influenced_by))
    plt.title(title_str , fontsize=10)
    plt.show()
    # Plot grid of images
    fig, axes = plt.subplots(1,10, figsize=(20, 20))  # 5 rows, 2 columns
    for i, ax in enumerate(axes.flatten()):
        if i < len(positive_indexes):
            image_path = general_image_path + df.iloc[positive_indexes[i]].relative_path
            image = Image.open(image_path)
            ax.imshow(image)
            ax.axis('off')
            if dataset_name == 'wikiart':
                title_str = (r"$\bf{rank:}$ " + str(i+1) + '\n'
                r"$\bf{Agent:}$ " +df.loc[positive_indexes[i]].artist_name +'\n' + r"$\bf{date:}$ " + str(df.loc[positive_indexes[i]].date_filled))
            else:
                title_str = (r"$\bf{rank:}$ " + str(i+1) + '\n'
                r"$\bf{Agent:}$ " +df.loc[positive_indexes[i]].artist_name)

            ax.set_title(title_str, fontsize = 10)
        else:
            ax.axis('off')  # Hide unused subplots

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()



def print_mean_metrics(IR):
        print(f'Pk: {round(np.mean(list(IR["precision_at_k_artist"].values())),3)}, MRR: {round(np.mean(list(IR["mrr_artist"].values())),3)}')
        print(f'2Pk: {round(np.mean(list(IR["precision_at_k_artist_second_degree"].values())),3)}, 2MRR: {round(np.mean(list(IR["mrr_artist_second_degree"].values())),3)}')


def metrics(IR):
        print(f'Precision at k10 for artist: {IR["precision_at_k_mean"]}, MRR for artist: {round(np.mean(list(IR["mrr_artist"].values())),3)}')
        print(f'Precision at k10 for second degree artist: {IR["precision_at_k_second_degree_mean"]}, MRR for second degree artist: {IR["mrr_second_degree_mean"]}')

def print_metrics(dataset_name, viz = True):
    if dataset_name == "wikiart":
        if viz:
            df = pd.read_pickle('DATA/Dataset/wikiart/wikiart_full_combined_no_artist_filtered.pkl')
            index = 2896
        features = ["image_features", "image_text_features"]

    if dataset_name == "fashion":
        if viz:
            df = pd.read_pickle('DATA/Dataset/iDesigner/idesigner_influences_cropped_features_mode.pkl')
            index = 3199
        features = ["image_features"]
    feature_extractors = ["ResNet34_newsplit","random_artists"]
    sampling_strategies = ["posrandom", "posfaiss"]
    num_examples = ["10", "100"]
    for feature in features:
        for feature_extractor in feature_extractors:
            if viz:
                if dataset_name == "wikiart" and feature_extractor == "ResNet34_newsplit" :
                    df = split_by_strata_artist(df) 
                elif dataset_name == "fashion" and feature_extractor == "ResNet34_newsplit" :
                    df = split_by_strata_artist_designer(df)
                    i = 16
                elif feature_extractor == "random_artists":
                    df = split_by_artist_random(df)
            path = f'trained_models/{dataset_name}/{feature_extractor}/baseline_IR_metrics/{feature}_val.pth'
            IR_baseline = torch.load(path)
            print(f'BASELINE with {feature_extractor}, {feature}')
            metrics(IR_baseline)
            print('     ')
            for num_example in num_examples:
                for sampling_strategy in sampling_strategies:
                    path = f'trained_models/{dataset_name}/{feature_extractor}/TripletResNet_{feature}_{sampling_strategy}_negrandom_{num_example}_margin1_notrans_epoch_30/IR_metrics/metrics_val.pth'
                    IR_metrics = torch.load(path)
                    print(f'Experiment:{feature_extractor}, {feature}, {sampling_strategy} and {num_example} ')
                    metrics(IR_metrics)
                    if viz:
                        # if dataset_name == "wikiart":
                        #     indices = df[(df['mode'] == 'val') & (df['artist_name'] == 'vincent-van-gogh')].index.tolist()
                        # else: 
                        #     indices = df[(df['mode'] == 'val') & (df['artist_name'] == 'alexander mcqueen')].index.tolist()
                        plot_examples(dataset_name,index,IR_metrics['retrieved_indexes'][index], df)
                    print('    ')
            print('----------')

def correlation_number_positive_precision(dict_len_positive, dict_precision):
        values1 = [dict_len_positive[key] for key in dict_len_positive]
        values2 = [dict_precision[key] for key in dict_precision]

        # Calculate Pearson correlation coefficient
        correlation, _ = pearsonr(values1, values2)
        return correlation