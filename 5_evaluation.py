import faiss
import numpy as np
import time
import torch
import pandas as pd 
import random
from Triplet_Network import TripletResNet_features
import torch.nn as nn
import torch
import os
import argparse
from functions import split_by_artist_given, split_by_strata_artist, split_by_strata_artist_designer, split_by_artist_random, split_based_on_popularity

class Evaluation():
    def __init__(self,dataset_name,df,feature,device,mode):
        self.keep_without_influencers = False
        self.dataset_name = dataset_name
        self.feature = feature
        self.device = device
        self.df = df
        self.df_mode = self.df[self.df['mode'] == mode].copy()
        self.dict_influence_indexes, self.artist_to_paintings, self.dict_influenced_by = self.get_dictionaries()

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

        


    def get_dictionaries(self):
        dict_influenced_by = self.df_mode.groupby('artist_name')['influenced_by'].first().to_dict()
        artist_to_paintings = {}
        for index, row in self.df.iterrows():
            artist = row['artist_name']
            artist_to_paintings.setdefault(artist, []).append(index)

        artist_to_influencer_paintings = {
            artist: [
                painting for influencer in influencers if influencer in artist_to_paintings 
                for painting in artist_to_paintings[influencer]
            ] 
            for artist, influencers in dict_influenced_by.items()
        }
        return artist_to_influencer_paintings, artist_to_paintings, dict_influenced_by

    def vector_similarity_search_group(self,query_indexes, index_list):
        '''Search for similar vectors in the dataset using faiss library'''
        k = 10

        xb = torch.stack(self.df.loc[index_list, self.feature].tolist())
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
        for query in query_indexes:
            query_vector = self.df[self.feature].tolist()[query]
            D, I = index.search(query_vector.reshape(1,-1), k)  
            I = [index_list[i] for i in list(I[0])]
            self.df_mode.at[query, f'pos_ex_{self.feature}'] = I
        return self.df_mode[f'pos_ex_{self.feature}']



    def retrieval(self):
        grouped = self.df_mode.groupby('artist_name')
        self.df_mode.loc[:,f'pos_ex_{self.feature}'] = [None]*len(self.df_mode)
        for artist, group in grouped:
            queries = [i for i in list(group.index)]
            artist_index_list = self.artist_to_paintings[artist]
            #if agent has as influencer only himself (mostly for idesigner)
            if self.df_mode[self.df_mode['artist_name']==artist].reset_index().influenced_by[0][0] == artist and len(self.df_mode[self.df_mode['artist_name']==artist].reset_index().influenced_by[0])==1:
                index_list = list(set(self.df.index.tolist()))
            else:
                index_list = list(set(self.df.index.tolist()) - set(artist_index_list))
            if self.dataset_name == "wikiart": #not consider objects with date higher than the max of the artist grouped
                index_date_higher = self.df[self.df['date_filled'] >= int(group['date_filled'].max())].index.tolist()
                index_list = sorted(list(set(index_list) - set(index_date_higher)))
            self.df_mode[f'pos_ex_{self.feature}'] = self.vector_similarity_search_group(queries, index_list)
        return self.df_mode[f'pos_ex_{self.feature}']
    
    def metrics_per_query(self, queries, results, influencers_list):
        p_10_sum, mrr_sum = 0,0
        precision_dict_list = []
        for i in queries:
            p_10 = len([j for j in results[i] if j in influencers_list])/10
            dict_results = {k: results[i][:k] for k in range(1, len(results[i]) + 1)}
            dict_influencers_in_results = {key: [value for value in values if value in influencers_list] for key, values in dict_results.items()}
            precision_dict = {k: len(v)/k for k,v in dict_influencers_in_results.items()}

            mrr = self.mean_reciprocal_rank({i:influencers_list}, {i:results[i]})
            p_10_sum += p_10
            mrr_sum += mrr
            precision_dict_list.append(precision_dict)
        
        return p_10_sum,precision_dict_list,mrr_sum
    
    def evaluate_retrieval(self):
        precision_at_k_sum , precision_at_k_second_degree_sum, mrr_sum_,mrr_second_degree_sum= 0,0,0,0
        precision_at_k_artist, precision_at_k_artist_second_degree = {}, {}
        precisions_dict_result, precisions_dict_result_second_degree = {}, {}
        mrr_artist, mrr_artist_second_degree = {}, {}

        grouped = self.df_mode.groupby('artist_name')
        self.df_mode[f'pos_ex_{self.feature}'] = self.retrieval()

        for artist, group in grouped:
            queries = [i for i in list(group.index)]
            #ground truth
            influencers_list = self.dict_influence_indexes[artist]
            artist_influencers = [j for j in self.dict_influenced_by[artist] if j in self.dict_influence_indexes.keys()]
            second_degree_influencers = [self.dict_influence_indexes[j] for j in artist_influencers]
            if len(second_degree_influencers) > 0:
                second_degree_influencers = second_degree_influencers[0] + influencers_list
            else:
                second_degree_influencers = influencers_list

            #retrieved metrics
            p_10_sum, precision_dict_list,mrr_sum= self.metrics_per_query(queries, self.df_mode[f'pos_ex_{self.feature}'],influencers_list )
            p_10_sum_second,  precision_dict_list_second,mrr_sum_second = self.metrics_per_query(queries, self.df_mode[f'pos_ex_{self.feature}'],second_degree_influencers )
            precision_at_k_sum += p_10_sum
            precision_at_k_second_degree_sum += p_10_sum_second
            mrr_sum_ += mrr_sum   
            mrr_second_degree_sum += mrr_sum_second          
            precision_at_k_artist[artist],precision_at_k_artist_second_degree[artist]  = round(p_10_sum/len(queries),3), round(p_10_sum_second/len(queries),3)
            mrr_artist[artist], mrr_artist_second_degree[artist] = round(mrr_sum/len(queries),3), round(mrr_sum_second/len(queries),3)

            #precision at different k
            precisions_dict_result[artist] = {key: round(sum(d[key] for d in precision_dict_list) / len(precision_dict_list),3) for key in precision_dict_list[0]}
            precisions_dict_result_second_degree[artist] = {key: round(sum(d[key] for d in precision_dict_list_second) / len(precision_dict_list_second),3) for key in precision_dict_list_second[0]}

 
        precision_at_k_mean = round(precision_at_k_sum/len(self.df_mode),3)
        precision_at_k_second_degree_mean = round(precision_at_k_second_degree_sum/len(self.df_mode),3)
        mrr_mean = round(mrr_sum_/len(self.df_mode),3)
        mrr_second_degree_mean = round(mrr_second_degree_sum/len(self.df_mode),3)

        print(f'Precision at k10 for artist: {precision_at_k_mean}, MRR for artist: {mrr_mean}')
        print(f'Precision at k10 for second degree artist: {precision_at_k_second_degree_mean}, MRR for second degree artist: {mrr_second_degree_mean}')
        print('---------------------------------------')
        print(' ')

        IR_metrics = { 'retrieved_indexes': self.df_mode[f'pos_ex_{self.feature}'], 'precision_at_k_artist': precision_at_k_artist, 'mrr_artist': mrr_artist, 'precision_at_k_artist_second_degree': precision_at_k_artist_second_degree, 'mrr_artist_second_degree': mrr_artist_second_degree, 'precisions_dict_result': precisions_dict_result, 'precisions_dict_result_second_degree': precisions_dict_result_second_degree,'precision_at_k_mean' : precision_at_k_mean , 'precision_at_k_second_degree_mean': precision_at_k_second_degree_mean, 'mrr_mean': mrr_mean,'mrr_second_degree_mean': mrr_second_degree_mean}
        return IR_metrics 


    





def main(dataset_name, feature,data_split, num_examples,positive_based_on_similarity, negative_based_on_similarity,artist_splits):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    mode = 'val'
    how_feature_positive = 'posfaiss' if positive_based_on_similarity else 'posrandom'
    how_feature_negative = 'negfaiss' if negative_based_on_similarity else 'negrandom'
    margin = 1
    if dataset_name == 'wikiart':
        if 'clip' in feature:
            df = pd.read_pickle('DATA/Dataset/wikiart/wikiartINFL_clip.pkl')
        else:
            df = pd.read_pickle('DATA/Dataset/wikiart/wikiartINFL.pkl')
        if data_split == "stratified_artists":
            df = split_by_strata_artist(df)
    elif dataset_name == 'fashion':
        if 'clip' in feature:
            df = pd.read_pickle('DATA/Dataset/iDesigner/idesignerINFL_clip.pkl')
        else:
            df = pd.read_pickle('DATA/Dataset/iDesigner/idesignerINFL.pkl')
        if data_split == "stratified_artists":
            df = df

    if data_split == "random_artists":
        df = split_by_artist_random(df)
    if data_split == "popular_artists":
        df = split_based_on_popularity(df)

    model = TripletResNet_features(df.loc[0,feature].shape[0])

    print(f'BASELINE METRIC with {feature}')
    if os.path.exists(f'trained_models/{dataset_name}/{data_split}/baseline_IR_metrics') == False:
        os.makedirs(f'trained_models/{dataset_name}/{data_split}/baseline_IR_metrics')
    IR_metrics_baseline = Evaluation(dataset_name, df,feature,device,mode).evaluate_retrieval()
    torch.save(IR_metrics_baseline,f'trained_models/{dataset_name}/{data_split}/baseline_IR_metrics/{feature}_{mode}.pth')


    trained_model_path = f'trained_models/{dataset_name}/{data_split}/TripletResNet_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}_margin{margin}_notrans_epoch_30'
    print(f'Features with model {trained_model_path}')
    model_path = trained_model_path + '/model.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    df[f'trained_{feature}'] = df[feature].apply(lambda x: model.forward_once(x).detach())
    IR_metrics = Evaluation(dataset_name,df,f'trained_{feature}',device,mode).evaluate_retrieval()
    if os.path.exists(f'{trained_model_path}/IR_metrics') == False:
        os.makedirs(f'{trained_model_path}/IR_metrics')
    torch.save(IR_metrics, f'{trained_model_path}/IR_metrics/metrics_{mode}.pth')

if __name__ == '__main__':
    start_time = time.time() 
    parser = argparse.ArgumentParser(description="Evaluation of the model under IR task")
    parser.add_argument('--dataset_name', type=str, default='wikiart', choices=['wikiart', 'fashion'])
    parser.add_argument('--feature', type=str, default='image_features', help='image_features text_features image_text_features')
    parser.add_argument('--artist_splits', action='store_true',help= 'create dataset excluding a gievn artist from training set' )
    parser.add_argument('--data_split', type=str, default = 'stratified_artists', help= ['stratified_artists', 'random_artists' 'popular_artists'])
    parser.add_argument('--num_examples', type=int, default=10, help= 'How many examples for each anchor')
    parser.add_argument('--positive_based_on_similarity',action='store_true',help='Sample positive examples based on vector similarity or randomly')
    parser.add_argument('--negative_based_on_similarity', action='store_true',help='Sample negative examples based on vector similarity or randomly')
    args = parser.parse_args()
    main(args.dataset_name,args.feature,args.data_split, args.num_examples,args.positive_based_on_similarity, args.negative_based_on_similarity, args.artist_splits)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required to extract the features: {:.2f} seconds".format(elapsed_time))
