import torch
from torch.utils.data import Dataset, DataLoader
import faiss
import random
import pandas as pd
import numpy as np
import time
import os
import argparse
from collections import Counter


os.environ['KMP_DUPLICATE_LIB_OK']='True'


class TripletLossDataset_features(Dataset):
    def __init__(self, mode, df, num_examples,feature,device, positive_similarity_based, negative_similarity_based):
        self.mode = mode
        self.feature = feature
        self.num_examples = num_examples
        self.device = device
        self.positive_similarity_based = positive_similarity_based
        self.negative_similarity_based = negative_similarity_based
        self.df = df[df['mode'] == mode].reset_index(drop=True)
        self.df = self.remove_influencers_without_examples(self.df)
        self.dict_influence_indexes, self.painter_indexes = self.get_dictionaries()
        self.filtered_indices = self.filter_indices()
        self.dimension = self.df.loc[0,self.feature].shape[0]  
        self.positive_examples = self.positive_examples_group()   
        self.negative_examples = self.get_negative_examples()  
        print('length of df:',len(self.df))

   
    def filter_indices(self):
        '''Filter indices based on mode and number of examples per anchor'''
        filtered = [index for index in range(len(self.df))]
        return filtered


    def remove_influencers_without_examples(self,df):
        all_artist_names = set(df['artist_name'])
        df['influenced_by'] = df['influenced_by'].apply(lambda artists_list: [artist for artist in artists_list if artist in all_artist_names])
        df = df[df['influenced_by'].apply(len)>0].reset_index(drop=True)
        return df

    def get_dictionaries(self):
        dict_influenced_by = self.df.groupby('artist_name')['influenced_by'].first().to_dict()
        artist_to_paintings = {}
        for index, row in self.df.iterrows():
            artist = row['artist_name']
            artist_to_paintings.setdefault(artist, []).append(index)
        artist_to_influencer_paintings = {artist: [painting for influencer in influencers if influencer in artist_to_paintings for painting in artist_to_paintings[influencer]] for artist, influencers in dict_influenced_by.items()}
        return artist_to_influencer_paintings, artist_to_paintings
    

    def vector_similarity_search_group(self,query_indexes, index_list):
        '''Search for similar vectors in the dataset using faiss library'''
        k = self.num_examples + 1

        index_list = [i for i in index_list if i < len(self.df)]
        xb = torch.stack(self.df[self.feature].tolist())[index_list]
        d = xb.shape[1]
        index = faiss.IndexFlatL2(d)
        if xb.shape[0] < 1000:
            index = faiss.IndexFlatL2(d)
        else:
            nlist = 10  
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
        
        if self.device == 'cuda':
            index = faiss.index_cpu_to_all_gpus(index)
            
        index.train(xb.cpu().numpy() if self.device != 'cuda' else xb)
        index.add(xb.cpu().numpy() if self.device != 'cuda' else xb)
        results = []
        for query in query_indexes:
            query_vector = self.df[self.feature].tolist()[query]
            D, I = index.search(query_vector.reshape(1,-1), k)  
            I = list(I[0][1:])   
            I = [index_list[i] for i in I]
            results.append(I)
            
        return results 



        
    def get_negative_examples(self):
        '''Get negative examples
        random_negative: if True, return random negative examples
         else return negative examples based on similarity to the anchor'''

        self.df[f'neg_ex_{self.feature}'] = [None]*len(self.df)
        grouped = self.df.groupby('artist_name')
        for artist, group in grouped:
            query = list(group.index)
            index_list = self.dict_influence_indexes[artist]
            artist_indexes = self.painter_indexes[artist]
            remaining_index_list = list(set(list(self.df.index)) - set(index_list) - set(artist_indexes)) 
            if self.negative_similarity_based:
                results = self.vector_similarity_search_group(query, remaining_index_list)
                for i,q in enumerate(query):
                    self.df.at[q,f'neg_ex_{self.feature}'] = results[i]
            else:
                for q in query:
                    self.df.at[q,f'neg_ex_{self.feature}'] = random.sample(remaining_index_list, self.num_examples)
        return self.df[f'neg_ex_{self.feature}']


    def positive_examples_group(self):
        '''Get positive examples for each anchor
        random_positive: if True, return N random positive examples, if there are less than N examples sample with replacement
        else return N positive examples based on similarity to the anchor'''

        grouped = self.df.groupby('artist_name')
        self.df[f'pos_ex_{self.feature}'] = [None]*len(self.df)
        for artist, group in grouped:
            query = list(group.index)
            query = [i for i in query]
            index_list = self.dict_influence_indexes[artist]
            if self.positive_similarity_based:
                if len(index_list) >= self.num_examples:
                    results = self.vector_similarity_search_group(query, index_list)
                    for i, q in enumerate(query):
                        self.df.at[q, f'pos_ex_{self.feature}'] = results[i]
                else:
                    for q in query:
                        self.df.at[q, f'pos_ex_{self.feature}'] = random.choices(index_list, k=self.num_examples)
            else:
                for q in query:
                    if len(index_list) >= self.num_examples:
                        self.df.at[q, f'pos_ex_{self.feature}'] = random.sample(index_list, self.num_examples)
                    else:
                        self.df.at[q, f'pos_ex_{self.feature}'] = random.choices(index_list, k=self.num_examples)
        return self.df[f'pos_ex_{self.feature}']

    
    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):

        index = self.filtered_indices[index]
        positive_indexes = self.positive_examples.iloc[index]
        negative_indexes = self.negative_examples.iloc[index]

        img_anchor = self.df[self.feature][index].repeat(self.num_examples,1)
        img_pos = torch.stack([self.df[self.feature][i] for i in positive_indexes])
        img_neg = torch.stack([self.df[self.feature][i] for i in negative_indexes])

        return img_anchor, img_pos, img_neg

from sklearn.model_selection import train_test_split
def split_by_strata_artist(df, train_size=0.7, val_size=0.25, test_size=0.05):
    df['mode'] = None
    
    for artist_name, group in df.groupby('artist_name'):
        train_indices, val_test_indices = train_test_split(group.index, train_size=train_size, random_state=42)
        val_indices, test_indices = train_test_split(val_test_indices, train_size=val_size/(val_size+test_size), random_state=42)
        
        df.loc[train_indices, 'mode'] = 'train'
        df.loc[val_indices, 'mode'] = 'val'
        df.loc[test_indices, 'mode'] = 'test'
    
    return df

def split_by_artist_random(df, train_size=0.7, val_size=0.25, test_size=0.05, random_state=55):
    unique_artists = df['artist_name'].unique()
    train_artists, remaining_artists = train_test_split(unique_artists, train_size=train_size, random_state=random_state)
    val_artists, test_artists = train_test_split(remaining_artists, train_size=val_size/(val_size+test_size), random_state=random_state)
    df['mode'] = None
    df.loc[df['artist_name'].isin(train_artists), 'mode'] = 'train'
    df.loc[df['artist_name'].isin(val_artists), 'mode'] = 'val'
    df.loc[df['artist_name'].isin(test_artists), 'mode'] = 'test'
    
    return df

def split_by_artist_given(df, artist_name):
    df['mode'] = None
    df.loc[df['artist_name'] != artist_name, 'mode'] = 'train'
    df.loc[df['artist_name'] == artist_name, 'mode'] = 'val'
    
    return df   

def main(dataset_name,feature,feature_extractor_name, num_examples, positive_based_on_similarity, negative_based_on_similarity):
    if dataset_name == 'wikiart':
        df = pd.read_pickle('DATA/Dataset/wikiart_full_combined_no_artist_filtered.pkl')
    elif dataset_name == 'fashion':
        df = pd.read_pickle('DATA/Dataset/iDesigner/idesigner_influences_cropped_features.pkl')
        
    if feature_extractor_name == "Random_artist_split":
        df = split_by_artist_random(df)
    elif feature_extractor_name == "ResNet34_newsplit":
        df = split_by_strata_artist(df)

   
    if os.path.exists(f'DATA/Dataset_toload/{dataset_name}/{feature_extractor_name}') == False:
        os.makedirs(f'DATA/Dataset_toload/{dataset_name}/{feature_extractor_name}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    how_feature_positive = 'posfaiss' if positive_based_on_similarity else 'posrandom'
    how_feature_negative = 'negfaiss' if negative_based_on_similarity else 'negrandom'
    train_dataset = TripletLossDataset_features('train', df, num_examples, feature, device, positive_based_on_similarity, negative_based_on_similarity)
    val_dataset = TripletLossDataset_features('val', df, num_examples, feature, device, positive_based_on_similarity, negative_based_on_similarity)
    torch.save(train_dataset, f'DATA/Dataset_toload/{dataset_name}/{feature_extractor_name}/train_dataset_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}.pt')
    torch.save(val_dataset, f'DATA/Dataset_toload/{dataset_name}/{feature_extractor_name}/val_dataset_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}.pt')


def main_artists(dataset_name,feature,feature_extractor_name, num_examples, positive_based_on_similarity, negative_based_on_similarity):
    if dataset_name == 'wikiart':
        df = pd.read_pickle('DATA/Dataset/wikiart_full_combined_no_artist_filtered.pkl')
    elif dataset_name == 'fashion':
        df = pd.read_pickle('DATA/Dataset/iDesigner/idesigner_influences_cropped_features.pkl')
    #artists = df['artist_name'].unique()
    artist = feature_extractor_name
    df = split_by_artist_given(df, artist)
    if os.path.exists(f'DATA/Dataset_toload/{dataset_name}/Artists') == False:
        os.makedirs(f'DATA/Dataset_toload/{dataset_name}/Artists')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    how_feature_positive = 'posfaiss' if positive_based_on_similarity else 'posrandom'
    how_feature_negative = 'negfaiss' if negative_based_on_similarity else 'negrandom'
    train_dataset = TripletLossDataset_features('train', df, num_examples, feature, device, positive_based_on_similarity, negative_based_on_similarity)
    torch.save(train_dataset, f'DATA/Dataset_toload/{dataset_name}/Artists/{artist}_train_dataset_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}.pt')

if __name__ == "__main__":
    start_time = time.time() 
    parser = argparse.ArgumentParser(description="Create dataset for triplet loss network on wikiart to predict influence.")
    parser.add_argument('--dataset_name', type=str, default='wikiart', choices=['wikiart', 'fashion'])
    parser.add_argument('--feature', type=str, default='image_features', help='image_features text_features image_text_features')
    parser.add_argument('--artist_splits', action='store_true',help= 'create dataset excluding a gievn artist from training set' )
    parser.add_argument('--feature_extractor_name', type=str, default = 'PabloPicasso', help= ['ResNet34', 'ResNet34_newsplit' 'ResNet152'])
    parser.add_argument('--num_examples', type=int, default=10, help= 'How many examples for each anchor')
    parser.add_argument('--positive_based_on_similarity',action='store_true',help='Sample positive examples based on vector similarity or randomly')
    parser.add_argument('--negative_based_on_similarity', action='store_true',help='Sample negative examples based on vector similarity or randomly')
    args = parser.parse_args()
    if args.artist_splits:
        main_artists(args.dataset_name,args.feature,args.feature_extractor_name, args.num_examples,args.positive_based_on_similarity, args.negative_based_on_similarity)
    else:
        main(args.dataset_name,args.feature,args.feature_extractor_name, args.num_examples,args.positive_based_on_similarity, args.negative_based_on_similarity)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required to build dataset: {:.2f} seconds".format(elapsed_time))