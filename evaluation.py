import faiss
import numpy as np
import time
import torch
import pandas as pd 
import random
from Triplet_Network import TripletResNet_features
import torch.nn as nn
import torch
from glob import glob
import os

class Evaluation():
    def __init__(self,df,feature,num_examples,device):
        self.df = df
        self.feature = feature
        self.num_examples = num_examples
        self.device = device
        self.dict_influence_indexes, self.artist_to_paintings, self.dict_influenced_by = self.get_dictionaries(df)

    def mean_reciprocal_rank(self,ground_truth, ranked_lists):
        """
        Compute the Mean Reciprocal Rank (MRR) for a set of queries.
        
        Parameters:
            ground_truth (dict): Dictionary where keys are query IDs and values are lists of relevant document IDs.
            ranked_lists (dict): Dictionary where keys are query IDs and values are ranked lists of document IDs.
            
        Returns:
            float: Mean Reciprocal Rank (MRR) value.
        """
        total_rr = 0.0
        num_queries = len(ground_truth)
        
        for query_id, relevant_docs in ground_truth.items():
            if query_id in ranked_lists:
                ranked_list = ranked_lists[query_id]
                reciprocal_rank = 0.0
                for i, doc_id in enumerate(ranked_list, start=1):
                    if doc_id in relevant_docs:
                        reciprocal_rank = 1.0 / i
                        break
                total_rr += reciprocal_rank
        
        if num_queries > 0:
            return total_rr / num_queries
        else:
            return 0.0  # Return 0 if there are no queries
        
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
        dict_influenced_by = {key: value for key, value in dict_influenced_by.items() if key in artist_to_influencer_paintings.keys()}

        return artist_to_influencer_paintings, artist_to_paintings_new, dict_influenced_by

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


    def positive_examples_group(self):
        search_among_influencers = False
        grouped = self.df.groupby('artist_name')
        self.df[f'pos_ex_{self.feature}'] = [None]*len(self.df)
        precision_at_k_artist, precision_at_k_artist_second_degree = {}, {}
        mrr_artist, mrr_artist_second_degree = {}, {}
        for artist, group in grouped:
            query = list(group.index)
            query = [i for i in query if i < len(self.df)]
            if artist in self.dict_influence_indexes:
                influencers_list = self.dict_influence_indexes[artist]
                artist_influencers = [j for j in self.dict_influenced_by[artist] if j in self.dict_influence_indexes.keys()]
                second_degree_influencers = [self.dict_influence_indexes[j] for j in artist_influencers]
                if len(second_degree_influencers) > 0:
                    second_degree_influencers = second_degree_influencers[0] + influencers_list
                else:
                    second_degree_influencers = influencers_list

                if search_among_influencers:
                    index_list = self.dict_influence_indexes[artist]
                    index_list = [i for i in index_list if i < len(self.df)]
                else:
                    no_index_list = self.artist_to_paintings[artist]
                    index_list = set(self.df.index) - set(no_index_list)

                if len(index_list) > 0:  # Check if index_list is not empty
                    results = self.vector_similarity_search_group(query, index_list, self.df)
                    precision_at_k, precision_at_k_second_degree = [], []
                    mrr_overall, mrr_overall_second_degree = [], []
                else:
                    results = []
                    precision_at_k, precision_at_k_second_degree = [], []
                    mrr_overall, mrr_overall_second_degree = [], []

                for i,q in enumerate(query):
                    self.df.at[q,f'pos_ex_{self.feature}'] = results[i]
                    influencers_in_results = [j for j in results[i] if j in influencers_list]
                    second_degree_influencers_in_results = [j for j in results[i] if j in second_degree_influencers]
                    mrr = self.mean_reciprocal_rank({q:influencers_list}, {q:results[i]})
                    mrr_second_degree = self.mean_reciprocal_rank({q:second_degree_influencers}, {q:results[i]})

                    precision_at_k.append(len(influencers_in_results)/self.num_examples)
                    mrr_overall.append(mrr)

                    precision_at_k_second_degree.append(len(second_degree_influencers_in_results)/self.num_examples)
                    mrr_overall_second_degree.append(mrr_second_degree)


                #influencers_right_artist = [len(i) for i in influencers_right_artist]
                print(f'For Artist {artist}, Influencers in results: {np.mean(precision_at_k)}, MRR: {np.mean(mrr_overall)}')
                print(f'For Artist {artist}, Second Degree Influencers in results: {np.mean(precision_at_k_second_degree)}, MRR: {np.mean(mrr_overall_second_degree)}')
                precision_at_k_artist[artist] = np.mean(precision_at_k)
                mrr_artist[artist] = np.mean(mrr_overall)
                precision_at_k_artist_second_degree[artist] = np.mean(precision_at_k_second_degree)
                mrr_artist_second_degree[artist] = np.mean(mrr_overall_second_degree)


        return self.df[f'pos_ex_{self.feature}'], precision_at_k_artist, mrr_artist, precision_at_k_artist_second_degree, mrr_artist_second_degree
    



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    df = pd.read_pickle('DATA/Dataset/wikiart_full_combined_try.pkl')
    unique_values = df['artist_name'].explode().unique()
    df['influenced_by'] = df['influenced_by'].apply(lambda x: [i for i in x if i in unique_values])
    df = df[df['influenced_by'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
    features = ['image_features', 'text_features', 'image_text_features']
    for feature in features:

        retrieved_indexes, precision_at_k_artist, mrr_artist,precision_at_k_artist_second_degree, mrr_artist_second_degree = Evaluation(df,feature,10,device).positive_examples_group()
        print(f'BASELINE METRIC with {feature}')
        print(f'Precision at k for artist: {np.mean(list(precision_at_k_artist.values()))}, MRR for artist: {np.mean(list(mrr_artist.values()))}')
        print(f'Precision at k for second degree artist: {np.mean(list(precision_at_k_artist_second_degree.values()))}, MRR for second degree artist: {np.mean(list(mrr_artist_second_degree.values()))}')
        print('---------------------------------------')
        model = TripletResNet_features(df.loc[0,feature].shape[0])
        trained_models_path = glob('trained_models/*', recursive = True)
        for i in trained_models_path:
            if 'TripletResNet_'+feature in i:
                print(f'Features with model {i}')
                model_path = i + '/model.pth'
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                df[f'trained_{i}'] = df[feature].apply(lambda x: model.forward_once(x).detach())
                retrieved_indexes, precision_at_k_artist, mrr_artist,precision_at_k_artist_second_degree, mrr_artist_second_degree = Evaluation(df,f'trained_{i}',10,device).positive_examples_group()
                IR_metrics = {'precision_at_k_artist': precision_at_k_artist, 'mrr_artist': mrr_artist, 'precision_at_k_artist_second_degree': precision_at_k_artist_second_degree, 'mrr_artist_second_degree': mrr_artist_second_degree}
                if os.path.exists(f'{i}/IR_metrics') == False:
                    os.makedirs(f'{i}/IR_metrics')
                torch.save(IR_metrics, f'{i}/IR_metrics/metrics.pth')
                print(f'Precision at k for artist: {np.mean(list(precision_at_k_artist.values()))}, MRR for artist: {np.mean(list(mrr_artist.values()))}')
                print(f'Precision at k for second degree artist: {np.mean(list(precision_at_k_artist_second_degree.values()))}, MRR for second degree artist: {np.mean(list(mrr_artist_second_degree.values()))}')
                print('---------------------------------------')

if __name__ == '__main__':
    start_time = time.time() 
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required to extract the features: {:.2f} seconds".format(elapsed_time))
