import torch
import ast
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import faiss
import random
import itertools
from collections import Counter
import os
import time
import faiss.contrib.torch_utils

class TripletLossDataset_features(Dataset):
    def __init__(self, mode, df, num_examples,feature,device):
        self.df = df
        self.mode = mode
        self.feature = feature
        self.num_examples = num_examples
        self.device = device
        self.positive_examples = self.min_val()
        self.filtered_indices = self.filter_indices()
        self.dimension = self.df[self.feature][0].shape[0]       
        print('Number of observations after filtering:',len(self.filtered_indices))
   

    def filter_indices(self):
        '''Filter indices based on mode and number of examples per anchor'''
        return [index for index in range(len(self.df)) if self.df.iloc[index]['mode'] == self.mode and index in list(set(tup[0] for tup in self.positive_examples))]

    def get_positive_example_indices(self):
        '''List of tuples with positive examples (influenced, influencial_artist)'''

        positive_examples = []
        unique_values = self.df['artist_name'].explode().unique()

        for value in unique_values:
            mask_influence = self.df['influenced_by'].apply(lambda x: value in str(x)) 
            mask_artist = self.df['artist_name'].apply(lambda x: value in str(x)) 
            influence_indexes = self.df.index[mask_influence].tolist()
            artist_indexes = self.df.index[mask_artist].tolist()
            positive_examples.extend((influence_index, artist_index) for influence_index, artist_index in itertools.product(influence_indexes, artist_indexes))

        return positive_examples
    
    def min_val(self):
        '''Filter positive examples to include only artists with more than n examples'''
        positive_examples_count = {}
        positive_examples = self.get_positive_example_indices()
        for artist_index, influence_index in positive_examples:
            positive_examples_count[artist_index] = positive_examples_count.get(artist_index, 0) + 1
        positive_examples_more = [(artist_index, influence_index) for artist_index, influence_index in positive_examples if positive_examples_count[artist_index] >= self.num_examples]

        return positive_examples_more
    
    def vector_similarity_search(self, query, index_list):
        '''Search for similar vectors in the dataset using faiss library'''
        k = self.num_examples+1
        if index_list != None:
            xb = torch.stack(self.df[self.feature].tolist())[index_list]
        else:
            xb = torch.stack(self.df[self.feature].tolist())
        d = xb.shape[1]
        # if self.device == 'mps':
        #     index = faiss.IndexFlatL2(d) 
        # else:
        #     index = faiss.GpuIndexFlatL2(d)
        quantizer = faiss.IndexFlatL2(d)
        nlist = 100
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        if self.device == 'cuda':
            index = faiss.index_cpu_to_all_gpus(index)
        index.train(xb) if self.device == 'cuda' else index.train(xb.cpu().numpy())
        index.add(xb) if self.device == 'cuda' else index.add(xb.cpu().numpy())

        D, I = index.search(xb[query].reshape(1,-1), k)  
        I = list(I[0][1:])   
        I = [index_list[i] for i in I]
        return I



    def get_negative_examples(self,x, random_negative=True):
        '''Get negative examples
        random_negative: if True, return random negative examples
         else return negative examples based on similarity to the anchor'''

        matching_examples = {example[1] for example in self.positive_examples if example[1] == x}
        remaining_indexes = list(set(self.filtered_indices) - matching_examples)
        if random_negative:
            return random.sample(remaining_indexes, min(self.num_examples, len(remaining_indexes)))
        else:
            return self.vector_similarity_search(x, remaining_indexes) #negative examples are similar to the anchor but not positive
  

    def get_positive_examples(self,x,based_on_features=True):
        '''Get positive examples
        based_on_features: if True, return positive examples based on similarity to the anchor
         else return random positive examples'''
        matching_indexes = [example[1] for example in self.positive_examples if example[0] == x]
        #influencers = {self.df.loc[idx, 'artist_name'] for idx in matching_indexes}

        if based_on_features:
            positive_indexes = self.vector_similarity_search(x,matching_indexes)
            return positive_indexes
        else:
            influencer_counts = Counter(self.df.loc[idx, 'artist_name'] for idx in matching_indexes)
            sampled_indexes = []
            for influencer, count in influencer_counts.items():
                indexes = [idx for idx in matching_indexes if self.df.loc[idx, 'artist_name'] == influencer]
                sampled_indexes.extend(random.choices(indexes, k=min(count, self.num_examples // len(influencer_counts))))
            return sampled_indexes[:self.num_examples]
            #return random.sample(matching_indexes, min(self.num_examples, len(matching_indexes)))
    
    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):

        index = self.filtered_indices[index]
        positive_indexes = self.get_positive_examples(index, based_on_features=True)
        negative_indexes = self.get_negative_examples(index, random_negative=True)       

        img_anchor = self.df[self.feature][index].repeat(self.num_examples,1)
        img_pos = torch.stack([self.df[self.feature][i] for i in positive_indexes])
        img_neg = torch.stack([self.df[self.feature][i] for i in negative_indexes])

        # img_anchor = [self.df[self.feature][index] for i in range(self.num_examples)]
        # img_pos = [self.df[self.feature][i] for i in positive_indexes]
        # img_neg = [self.df[self.feature][i] for i in negative_indexes]
        return img_anchor, img_pos, img_neg


def column_list_to_tensor(df,feature):
    df[feature] = df[feature].apply(lambda x: ast.literal_eval(x))
    df[feature] = df[feature].apply(lambda x: torch.tensor(x))
    return df[feature]

def main():
    dataset_path = 'DATA/Dataset/wikiart_full_combined.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    df = pd.read_csv(dataset_path)
    feature = 'image_features'
    df = df[:5000]
    
    df[feature] = column_list_to_tensor(df,feature)
    tripleloss_dataset_train = TripletLossDataset_features(mode='train', df = df,num_examples = 10, feature=feature,device=device)
    tripleloss_dataset_val = TripletLossDataset_features(mode='val', df = df,num_examples = 10, feature=feature,device=device)
    if os.path.exists('DATA/Dataset_toload') == False:
        os.makedirs('DATA/Dataset_toload')

    torch.save(tripleloss_dataset_train, 'DATA/Dataset_toload/train_dataset_image_features.pt')
    torch.save(tripleloss_dataset_val, 'DATA/Dataset_toload/val_dataset_image_features.pt')

if __name__ == "__main__":
    start_time = time.time() 
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required to build dataset: {:.2f} seconds".format(elapsed_time))