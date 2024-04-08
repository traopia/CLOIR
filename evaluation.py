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
import argparse

class Evaluation():
    def __init__(self,dataset_name,df,feature,device,mode):
        self.try_df_mode = True
        self.df = self.remove_influencers_without_examples(df)
        if self.try_df_mode:
            self.df = df
            self.df_mode = df[df['mode'] == mode]
            self.df_mode = self.df_mode.copy()
        else:
            self.df = df[df['mode'] == mode].reset_index(drop=True)

        self.dataset_name = dataset_name
        self.feature = feature
        self.device = device
        self.dict_influence_indexes, self.artist_to_paintings, self.dict_influenced_by = self.get_dictionaries(df)

    def remove_influencers_without_examples(self,df):
        all_artist_names = set(df['artist_name'])
        df['influenced_by'] = df['influenced_by'].apply(lambda artists_list: [artist for artist in artists_list if artist in all_artist_names])
        df = df[df['influenced_by'].apply(len)>0].reset_index(drop=True)
        return df

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
        # keys_min_val = [key for key, value in artist_to_influencer_paintings.items() if isinstance(value, list) and len(value) > 10]
        # artist_to_influencer_paintings = {key: value for key, value in artist_to_influencer_paintings.items() if key in keys_min_val}
        # artisit_no_influencers = [k for k, v in artist_to_influencer_paintings.items() if len(v) == 0]
        # artist_to_influencer_paintings = {key: value for key, value in artist_to_influencer_paintings.items() if key not in artisit_no_influencers}
        # artist_to_paintings_new = {key: value for key, value in artist_to_paintings.items() if key in artist_to_influencer_paintings.keys()}
        # dict_influenced_by = {key: value for key, value in dict_influenced_by.items() if key in artist_to_influencer_paintings.keys()}

        return artist_to_influencer_paintings, artist_to_paintings, dict_influenced_by

    def vector_similarity_search_group(self,query_indexes, index_list,df):
        '''Search for similar vectors in the dataset using faiss library'''
        k = 10 + 1
        xb = torch.stack(df[self.feature].tolist())[index_list]
        #index_list = [i for i in index_list if i < len(self.df)]
        # if index_list != None:
        #     xb = torch.stack(df[self.feature].tolist())[index_list]
        # else:
        #     xb = torch.stack(df[self.feature].tolist())
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
        if self.try_df_mode:
            grouped = self.df_mode.groupby('artist_name')
            self.df_mode.loc[:,f'pos_ex_{self.feature}'] = [None]*len(self.df_mode)
        else:
            grouped = self.df.groupby('artist_name')
            self.df.loc[:,f'pos_ex_{self.feature}'] = [None]*len(self.df)
        precision_at_k_artist, precision_at_k_artist_second_degree = {}, {}
        precisions_dict_result, precisions_dict_result_second_degree = {}, {}
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
                    if self.dataset_name == "wikiart":
                        index_date_higher = self.df[self.df['date'] > group.date.mean() ].index.tolist()
                        index_list = list(set(self.df.index) - set(no_index_list) - set(index_date_higher))
                    elif self.dataset_name == "fashion":
                        if self.df[self.df['artist_name']==artist].reset_index().influenced_by[0][0] == artist:
                            index_list = list(set(self.df.index))
                        else:
                            index_list = list(set(self.df.index) - set(no_index_list))
                    

                if len(index_list) > 0:  # Check if index_list is not empty
                    results = self.vector_similarity_search_group(query, index_list, self.df)
                    precision_at_k, precision_at_k_second_degree = [], []
                    precisions_dict, precisions_dict_second_degree = [], []
                    mrr_overall, mrr_overall_second_degree = [], []
                else:
                    results = []
                    precision_at_k, precision_at_k_second_degree = [], []
                    precisions_dict, precisions_dict_second_degree = [], []
                    mrr_overall, mrr_overall_second_degree = [], []

                for i,q in enumerate(query):
                    if self.try_df_mode:
                        self.df_mode.at[q,f'pos_ex_{self.feature}'] = results[i]
                    else:
                        self.df.at[q,f'pos_ex_{self.feature}'] = results[i]
                    influencers_in_results = [j for j in results[i] if j in influencers_list]
                    dict_results = {k: results[i][:k] for k in range(1, len(results[i]) + 1)}
                    dict_influencers_in_results = {key: [value for value in values if value in influencers_list] for key, values in dict_results.items()}
                    precision_dict = {k: len(v)/k for k,v in dict_influencers_in_results.items()}

                    second_degree_influencers_in_results = [j for j in results[i] if j in second_degree_influencers]
                    dict_second_degree_influencers_in_results = {key: [value for value in values if value in second_degree_influencers] for key, values in dict_results.items()}
                    precision_dict_second_degree = {k: len(v)/k for k,v in dict_second_degree_influencers_in_results.items()}

                    mrr = self.mean_reciprocal_rank({q:influencers_list}, {q:results[i]})
                    mrr_second_degree = self.mean_reciprocal_rank({q:second_degree_influencers}, {q:results[i]})

                    precision_at_k.append(len(influencers_in_results)/10)
                    mrr_overall.append(mrr)
                    precisions_dict.append(precision_dict)

                    precision_at_k_second_degree.append(len(second_degree_influencers_in_results)/10)
                    mrr_overall_second_degree.append(mrr_second_degree)
                    precisions_dict_second_degree.append(precision_dict_second_degree)

                precision_at_k_artist[artist] = np.mean(precision_at_k)
                mrr_artist[artist] = np.mean(mrr_overall)
                precision_at_k_artist_second_degree[artist] = np.mean(precision_at_k_second_degree)
                mrr_artist_second_degree[artist] = np.mean(mrr_overall_second_degree)

                precisions_dict_result[artist] = {key: sum(d[key] for d in precisions_dict) / len(precisions_dict) for key in precisions_dict[0]}
                precisions_dict_result_second_degree[artist] = {key: sum(d[key] for d in precisions_dict_second_degree) / len(precisions_dict_second_degree) for key in precisions_dict_second_degree[0]}
        print(f'Precision at k10 for artist: {np.mean(list(precision_at_k_artist.values()))}, MRR for artist: {np.mean(list(mrr_artist.values()))}')
        print(f'Precision at k10 for second degree artist: {np.mean(list(precision_at_k_artist_second_degree.values()))}, MRR for second degree artist: {np.mean(list(mrr_artist_second_degree.values()))}')
        print('Precision at different k:', {inner_key: sum(d[inner_key] for d in precisions_dict_result.values()) / len(precisions_dict_result) for inner_key in precisions_dict_result[next(iter(precisions_dict_result))].keys()})
        print('Precision at different k for second degree:', {inner_key: sum(d[inner_key] for d in precisions_dict_result_second_degree.values()) / len(precisions_dict_result_second_degree) for inner_key in precisions_dict_result_second_degree[next(iter(precisions_dict_result_second_degree))].keys()})
        print('---------------------------------------')
        print('                                       ')

        if self.try_df_mode:
            return self.df_mode[f'pos_ex_{self.feature}'], precision_at_k_artist, mrr_artist, precision_at_k_artist_second_degree, mrr_artist_second_degree, precisions_dict_result, precisions_dict_result_second_degree
        else:
            return self.df[f'pos_ex_{self.feature}'], precision_at_k_artist, mrr_artist, precision_at_k_artist_second_degree, mrr_artist_second_degree, precisions_dict_result, precisions_dict_result_second_degree

    


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

def split_by_artist_given(df, artist_name):
    df['mode'] = None
    df.loc[df['artist_name'] != artist_name, 'mode'] = 'train'
    df.loc[df['artist_name'] == artist_name, 'mode'] = 'val'
    
    return df   

def main(dataset_name,artist_splits,feature_extractor_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    if dataset_name == 'wikiart':
        df = pd.read_pickle('DATA/Dataset/wikiart_full_combined_no_artist_filtered.pkl')
    elif dataset_name == 'fashion':
        df = pd.read_pickle('DATA/Dataset/iDesigner/idesigner_influences_cropped_features.pkl')

    if artist_splits:
        artist_name = feature_extractor_name
        feature_extractor_name = 'Artists'
        df = split_by_artist_given(df, artist_name)
    if feature_extractor_name == "ResNet34_newsplit":
        df = split_by_strata_artist(df)
    mode = 'val'
    #df = df[df['mode'] == mode].reset_index(drop=True)
    features = ['image_features', 'text_features', 'image_text_features']
    features = ['image_features']
    for feature in features:
        if artist_splits:
            print(f'BASELINE METRIC with {feature} for  {artist_name}')
        else:
            print(f'BASELINE METRIC with {feature}')
        retrieved_indexes, precision_at_k_artist, mrr_artist,precision_at_k_artist_second_degree, mrr_artist_second_degree,precisions_dict_result, precisions_dict_result_second_degree= Evaluation(dataset_name, df,feature,device,mode).positive_examples_group()

        IR_metrics_baseline = { 'retrieved_indexes': retrieved_indexes, 'precision_at_k_artist': precision_at_k_artist, 'mrr_artist': mrr_artist, 'precision_at_k_artist_second_degree': precision_at_k_artist_second_degree, 'mrr_artist_second_degree': mrr_artist_second_degree, 'precisions_dict_result': precisions_dict_result, 'precisions_dict_result_second_degree': precisions_dict_result_second_degree}
        if os.path.exists(f'trained_models/{dataset_name}/{feature_extractor_name}/baseline_IR_metrics') == False:
            os.makedirs(f'trained_models/{dataset_name}/{feature_extractor_name}/baseline_IR_metrics')
        if artist_splits:
            torch.save(IR_metrics_baseline,f'trained_models/{dataset_name}/{feature_extractor_name}/baseline_IR_metrics/{artist_name}_{feature}.pth')
        else:
            torch.save(IR_metrics_baseline,f'trained_models/{dataset_name}/{feature_extractor_name}/baseline_IR_metrics/{feature}.pth')
        

        model = TripletResNet_features(df.loc[0,feature].shape[0])
 
        trained_models_path = glob(f'trained_models/{dataset_name}/{feature_extractor_name}/*', recursive = True)
        #trained_models_path = ['trained_models/TripletResNet_image_text_features_posrandom_negrandom_100_margin10']
        for i in trained_models_path:
            if (artist_splits and artist_name + '_TripletResNet_' + feature in i) or (not artist_splits and 'TripletResNet_' + feature in i):
            #if 'TripletResNet_'+feature in i:
            #if 'TripletResNet_' + feature in i and i.find('100') > i.find('TripletResNet_' + feature): #
                print(f'Features with model {i}')
                model_path = i + '/model.pth'
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                df[f'trained_{i}'] = df[feature].apply(lambda x: model.forward_once(x).detach())
                retrieved_indexes, precision_at_k_artist, mrr_artist,precision_at_k_artist_second_degree, mrr_artist_second_degree ,precisions_dict_result, precisions_dict_result_second_degree = Evaluation(dataset_name,df,f'trained_{i}',device,mode).positive_examples_group()
                IR_metrics = { 'retrieved_indexes': retrieved_indexes, 'precision_at_k_artist': precision_at_k_artist, 'mrr_artist': mrr_artist, 'precision_at_k_artist_second_degree': precision_at_k_artist_second_degree, 'mrr_artist_second_degree': mrr_artist_second_degree, 'precisions_dict_result': precisions_dict_result, 'precisions_dict_result_second_degree': precisions_dict_result_second_degree}

            

                if os.path.exists(f'{i}/IR_metrics') == False:
                    os.makedirs(f'{i}/IR_metrics')
                torch.save(IR_metrics, f'{i}/IR_metrics/metrics_{mode}.pth')

if __name__ == '__main__':
    start_time = time.time() 
    parser = argparse.ArgumentParser(description="Evaluation of the model under IR task")
    parser.add_argument('--dataset_name', type=str, default='wikiart', choices=['wikiart', 'fashion'])
    parser.add_argument('--artist_splits', action='store_true',help= 'create dataset excluding a gievn artist from training set' )
    parser.add_argument('--feature_extractor_name', type=str, default='Artists', help='Name of the feature extractor model: Artists, ResNet34')
    args = parser.parse_args()
    main(args.dataset_name, args.artist_splits,args.feature_extractor_name)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required to extract the features: {:.2f} seconds".format(elapsed_time))
