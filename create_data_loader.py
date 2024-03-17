import torch
from torch.utils.data import Dataset, DataLoader
import faiss
import random
import pandas as pd
import numpy as np
import time
import os


os.environ['KMP_DUPLICATE_LIB_OK']='True'


class TripletLossDataset_features(Dataset):
    def __init__(self, mode, df, num_examples,feature,device):
        self.mode = mode
        self.feature = feature
        self.num_examples = num_examples
        self.device = device
        self.dict_influence_indexes, self.painter_indexes = self.get_dictionaries(df)
        self.filtered_indices = self.filter_indices(df)
        self.df = df #df[df.index.isin([value for sublist in self.dict_influence_indexes.values() for value in sublist])]#df[df.index.isin(self.filtered_indices)].reset_index(drop=True)
        self.dimension = self.df[self.feature][0].shape[0]  
        self.positive_examples = self.positive_examples_group(self.df)     
        print('Number of observations after filtering:',len(self.filtered_indices))
        print('length of df:',len(self.df))

   

    def filter_indices(self,df):
        '''Filter indices based on mode and number of examples per anchor'''
        all_values = [value for sublist in self.painter_indexes.values() for value in sublist]
        #filtered = [index for index in range(len(df)) if df.iloc[index]['artist_name'] in self.dict_influence_indexes.keys() ]# and index in all_values]
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
            #df[f'pos_ex_{self.feature}'].iloc[query] = I
            df.at[query,f'pos_ex_{self.feature}'] = I
            results.append(I)
            
            
        return df[f'pos_ex_{self.feature}']



        
    def get_negative_examples(self,x ):
        '''Get negative examples
        random_negative: if True, return random negative examples
         else return negative examples based on similarity to the anchor'''

        if self.df.loc[x,'artist_name'] in self.dict_influence_indexes:
            matching_examples = self.dict_influence_indexes[self.df.loc[x,'artist_name']]
            remaining_indexes = list(set(list(self.df.index)) - set(matching_examples))
            return random.sample(remaining_indexes, min(self.num_examples, len(remaining_indexes)))


  

    def positive_examples_group(self,df):
        
        grouped = df.groupby('artist_name')
        self.df[f'pos_ex_{self.feature}'] = [None]*len(self.df)
        for artist, group in grouped:
            query = list(group.index)
            query = [i for i in query if i < len(self.df)]
            if artist in self.dict_influence_indexes:
                index_list = self.dict_influence_indexes[artist]
                index_list = [i for i in index_list if i < len(self.df)]

            df[f'pos_ex_{self.feature}'] = self.vector_similarity_search_group(query, index_list,self.df)
        return df[f'pos_ex_{self.feature}']

    
    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        #if self.df.loc[index,'artist_name'] in self.dict_influence_indexes.keys():
            index = self.filtered_indices[index]
            
            positive_indexes = self.positive_examples.iloc[index]
            negative_indexes = self.get_negative_examples(index)

            
            #positive_indexes = self.positive_examples.iloc[index]
            #negative_indexes = self.get_negative_examples(index)  


            img_anchor = self.df[self.feature][index].repeat(self.num_examples,1)
            img_pos = torch.stack([self.df[self.feature][i] for i in positive_indexes])
            img_neg = torch.stack([self.df[self.feature][i] for i in negative_indexes])
            # img_anchor = [self.df[self.feature][index] for i in range(self.num_examples)]
            # img_pos = [self.df[self.feature][i] for i in positive_indexes]
            # img_neg = [self.df[self.feature][i] for i in negative_indexes]
            return img_anchor, img_pos, img_neg
        #else:
        #    print(index, self.df.loc[index,'artist_name'])
        #    pass


def main():
    df = pd.read_pickle('DATA/Dataset/wikiart_full_combined_try.pkl')
    unique_values = df['artist_name'].explode().unique()
    # df['influenced_by'] = df['influenced_by'].apply(lambda x: x.split(', '))
    df['influenced_by'] = df['influenced_by'].apply(lambda x: [i for i in x if i in unique_values])
    df = df[df['influenced_by'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
    print('Number of observations before filtering:',len(df))
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    feature = 'image_features'
    tripleloss_dataset_train = TripletLossDataset_features('train', df, 10, feature, device)
    val_dataset = TripletLossDataset_features('val', df, 10, feature, device)
    torch.save(tripleloss_dataset_train, f'DATA/Dataset_toload/train_dataset_{feature}_all.pt')
    torch.save(val_dataset, f'DATA/Dataset_toload/val_dataset_{feature}_all.pt')

if __name__ == "__main__":
    start_time = time.time() 
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required to build dataset: {:.2f} seconds".format(elapsed_time))