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
        self.dict_influence_indexes, self.painter_indexes = self.get_dictionaries(self.df)
        self.filtered_indices = self.filter_indices(self.df)
        self.dimension = self.df[self.feature][0].shape[0]  
        self.positive_examples = self.positive_examples_group(self.df)   
        self.negative_examples = self.get_negative_examples(self.df)  
        print('Number of observations after filtering:',len(self.filtered_indices))
        print('length of df:',len(self.df))

   

    def filter_indices(self,df):
        '''Filter indices based on mode and number of examples per anchor'''
        all_values = [value for sublist in self.painter_indexes.values() for value in sublist]
        filtered = [index for index in range(len(df)) if df.iloc[index]['mode'] == self.mode and index in all_values]
        for i in filtered:
            if df.iloc[i]['artist_name'] not in self.dict_influence_indexes.keys():
                print('not in dict',i,df.iloc[i]['artist_name'])
                filtered.remove(i)
        return filtered


    def get_dictionaries(self,df):
        dict_influenced_by = df.groupby('artist_name')['influenced_by'].first().to_dict()
        artist_to_paintings = {}
        for index, row in df.iterrows():
            artist = row['artist_name']
            artist_to_paintings.setdefault(artist, []).append(index)
        artist_to_influencer_paintings = {artist: [painting for influencer in influencers if influencer in artist_to_paintings for painting in artist_to_paintings[influencer]] for artist, influencers in dict_influenced_by.items()}
        keys_min_val = [key for key, value in artist_to_influencer_paintings.items() if isinstance(value, list) and len(value) > self.num_examples]
        artist_to_influencer_paintings = {key: value for key, value in artist_to_influencer_paintings.items() if key in keys_min_val}
        artisit_no_influencers = [k for k, v in artist_to_influencer_paintings.items() if len(v) == 0]
        artist_to_influencer_paintings = {key: value for key, value in artist_to_influencer_paintings.items() if key not in artisit_no_influencers}
        artist_to_paintings_new = {key: value for key, value in artist_to_paintings.items() if key in artist_to_influencer_paintings.keys()}
        return artist_to_influencer_paintings, artist_to_paintings_new

        
    def vector_similarity_search_group(self,query_indexes, index_list,df):
        '''Search for similar vectors in the dataset using faiss library'''
        k = self.num_examples + 1


        index_list = [i for i in index_list if i < len(self.df)]
        if index_list != None:
            xb = torch.stack(df[self.feature].tolist())[index_list]
        else:
            xb = torch.stack(df[self.feature].tolist())


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



        
    def get_negative_examples(self,df):
        '''Get negative examples
        random_negative: if True, return random negative examples
         else return negative examples based on similarity to the anchor'''

        self.df[f'neg_ex_{self.feature}'] = [None]*len(self.df)
        grouped = df.groupby('artist_name')
        for artist, group in grouped:
            query = list(group.index)
            if artist in self.dict_influence_indexes:
                index_list = self.dict_influence_indexes[artist]
                artist_indexes = self.painter_indexes[artist]
                remaining_index_list = list(set(list(self.df.index)) - set(index_list) - set(artist_indexes)) 
            if self.negative_similarity_based:
                results = self.vector_similarity_search_group(query, remaining_index_list,df)
                for i,q in enumerate(query):
                    self.df.at[q,f'neg_ex_{self.feature}'] = results[i]
            else:
                for q in query:
                    self.df.at[q,f'neg_ex_{self.feature}'] = random.sample(remaining_index_list, self.num_examples)
                
        return df[f'neg_ex_{self.feature}']


    def positive_examples_group(self,df):
        
        grouped = df.groupby('artist_name')
        self.df[f'pos_ex_{self.feature}'] = [None]*len(self.df)
        for artist, group in grouped:
            query = list(group.index)
            query = [i for i in query if i < len(self.df)]
            if artist in self.dict_influence_indexes:
                index_list = self.dict_influence_indexes[artist]
                index_list = [i for i in index_list if i < len(self.df)]
            if self.positive_similarity_based == True:
                results = self.vector_similarity_search_group(query, index_list,df)
                for i,q in enumerate(query):
                    self.df.at[q,f'pos_ex_{self.feature}'] = results[i]
                #self.df[f'pos_ex_{self.feature}'] = self.vector_similarity_search_group(query, index_list,self.df)
            else:
                for q in query:
                    self.df.at[q,f'pos_ex_{self.feature}'] = random.sample(index_list, self.num_examples)
        return df[f'pos_ex_{self.feature}']

    
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


def main(feature,num_examples, positive_based_on_similarity, negative_based_on_similarity):
    df = pd.read_pickle('DATA/Dataset/wikiart_full_combined_no_artist.pkl')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    how_feature_positive = 'posfaiss' if positive_based_on_similarity else 'posrandom'
    how_feature_negative = 'negfaiss' if negative_based_on_similarity else 'negrandom'
    train_dataset = TripletLossDataset_features('train', df, num_examples, feature, device, positive_based_on_similarity, negative_based_on_similarity)
    #val_dataset = TripletLossDataset_features('val', df, num_examples, feature, device, positive_based_on_similarity, negative_based_on_similarity)
    torch.save(train_dataset, f'DATA/Dataset_toload/train_dataset_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}.pt')
    #torch.save(val_dataset, f'DATA/Dataset_toload/val_dataset_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}.pt')

if __name__ == "__main__":
    start_time = time.time() 
    parser = argparse.ArgumentParser(description="Create dataset for triplet loss network on wikiart to predict influence.")
    parser.add_argument('--feature', type=str, default='image_features', help='image_features text_features image_text_features')
    parser.add_argument('--num_examples', type=int, default=10, help= 'How many examples for each anchor')
    parser.add_argument('--positive_based_on_similarity',action='store_true',help='Sample positive examples based on vector similarity or randomly')
    parser.add_argument('--negative_based_on_similarity', action='store_true',help='Sample negative examples based on vector similarity or randomly')
    args = parser.parse_args()
    main(args.feature, args.num_examples,args.positive_based_on_similarity, args.negative_based_on_similarity)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required to build dataset: {:.2f} seconds".format(elapsed_time))