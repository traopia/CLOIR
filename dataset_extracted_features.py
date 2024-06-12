import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import time 
import pickle
import os
import argparse
import clip
from torch.utils.data import DataLoader, Dataset

def calculate_average(s):
    parts = s.split('-')
    return (int(parts[0]) + int(parts[1])) / 2



def clean(df):
    df.dropna(subset=['influenced_by'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    unique_values = df['artist_name'].explode().unique()
    #df['influenced_by'] = df['influenced_by'].apply(lambda x: x.split(', '))
    df['influenced_by'] = df['influenced_by'].apply(lambda x: [i for i in x if i in unique_values])
    df = df[df['influenced_by'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

    all_artist_names = set(df['artist_name'])
    df['influenced_by'] = df['influenced_by'].apply(lambda artists_list: [artist for artist in artists_list if artist in all_artist_names])
    #drop if influenced by is empty
    df = df[df['influenced_by'].apply(len)>0].reset_index(drop=True)
    all_artist_names = set(df['artist_name'])
    df['influenced_by'] = df['influenced_by'].apply(lambda artists_list: [artist for artist in artists_list if artist in all_artist_names])
    df = df[df['influenced_by'].apply(len)>0].reset_index(drop=True)
    # Fill NaN values in 'col1' with the average of 'col2'
    df['date_filled'] = df.apply(lambda row: calculate_average(row['timeframe_estimation']) if pd.isna(row['date']) else row['date'], axis=1)
    return df


def image_features(image_path,general_path,device):
    #34 or 152
    weights = models.ResNet34_Weights.DEFAULT
    resnet = models.resnet34(weights=weights)
    resnet = resnet.to(device)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

    # Set model to evaluation mode
    resnet.eval()
    # Define image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_path = general_path + image_path
    image = Image.open(full_path)
    # Preprocess the image
    image_tensor = preprocess(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    # Move tensor to the selected device
    image_tensor = image_tensor.to(device)
    # Extract features
    with torch.no_grad():
        features = resnet(image_tensor)
    # Flatten the features
    features = features.squeeze().cpu()

    return features

def get_image_features(df,general_path, device):
    embeddings = []
    for image_path in df['relative_path']:
        embedding = image_features(image_path,general_path,device)
        embeddings.append(embedding)
    return embeddings

def get_embedding(text, model, tokenizer, device):
    # Tokenize the text
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        embedding = outputs.last_hidden_state.squeeze().mean(dim=0).cpu()
    return embedding

def get_text_features(df, device):
    # Load GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_basic_tokenize=False, do_lower_case=False)
    model = GPT2Model.from_pretrained('gpt2').to(device)
    df.loc[:, 'title']  = df.apply(lambda x: x.relative_path.split('/')[-1].split('.')[0].split('_')[1] if len(x.relative_path.split('/')[-1].split('.')[0].split('_')) ==2 else x.relative_path.split('/')[-1].split('.')[0].split('_')   , axis=1)
    df.loc[:, 'tags'] = df['tags'].fillna(' ')
    # Concatenate text from different columns
    df.loc[:,'concatenated_text'] = df['style_classification'].astype(str)  + ' '  + df['title'].astype(str) + ' ' + df['tags'].astype(str) + ' ' + df['timeframe_estimation'].astype(str) + ' ' + df['artist_school'].astype(str) 

    # Get embeddings for the concatenated text
    embeddings = []
    for text in df['concatenated_text']:
        embedding = get_embedding(text, model, tokenizer, device)
        embeddings.append(embedding)

    return embeddings



def preprocess_data(df,general_path,dataset_outpath, dataset_name, device):
    '''Preprocess the data and save it as a pickle file'''
    if dataset_name == 'wikiart':
        df['text_features'] = get_text_features(df,device)
        df['image_features'] = get_image_features(df,general_path, device)
        df['image_text_features'] = df.apply(lambda x: torch.cat([x['image_features'], x['text_features']]), axis=1)
    elif dataset_name == 'idesigner':
        df['image_features'] = get_image_features(df,general_path, device)

    df.to_pickle(dataset_outpath)

    return df

def clip_features(dataset_name, general_path, df, index,device,model, preprocess):


    # Load and preprocess the image
    image = preprocess(Image.open(general_path + df.loc[index, 'relative_path'])).unsqueeze(0)

    if dataset_name == "wikiart":
        # Tokenize the text metadata
        df.loc[:, 'title'] = df.apply(lambda x: x.relative_path.split('/')[-1].split('.')[0].split('_')[1] if len(x.relative_path.split('/')[-1].split('.')[0].split('_')) == 2 else x.relative_path.split('/')[-1].split('.')[0].split('_'), axis=1)
        #df.loc[:, 'tags'] = df['tags'].fillna(' ')
        text = clip.tokenize([str(df.loc[index, 'style_classification'] + ' ' + str(df.loc[index, 'timeframe_estimation']) + ' ' + str(df.loc[index, 'artist_school'])  + ' ' + str(df.loc[index, 'title']))])
        text = text.to(device)

    image = image.to(device)
    model = model.to(device)

    # Extract image embeddings
    with torch.no_grad():
        image_features = model.encode_image(image).cpu()
        if dataset_name == "wikiart":
            text_features = model.encode_text(text).cpu()

    # Normalize the embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)

    if dataset_name == "wikiart":
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features
    else:
        return image_features


def clip_process_data(dataset_name, general_path, df, dataset_outpath, device):
    indexes = df.index.tolist()
    df['clip_image_features'] = [None]*len(df)
    df['clip_text_features'] = [None]*len(df)
    model, preprocess = clip.load("ViT-B/32")
    model = model.to(device)
    for index in indexes:
        print(index)
        if dataset_name == "wikiart":
            image_features, text_features = clip_features(dataset_name, general_path, df, index,device,model,preprocess)
            df.at[index, 'clip_image_features'] = image_features
            df.at[index, 'clip_text_features'] = text_features

    df['clip_image_features'] = df['clip_image_features'].apply(lambda x: x.reshape(-1))
    df['clip_text_features'] = df['clip_text_features'].apply(lambda x: x.reshape(-1))
    df['clip_image_text_features'] = df.apply(lambda row: torch.cat((row['clip_image_features'], row['clip_text_features'])), axis=1)
    df.to_pickle(dataset_outpath)

    return df




def main(dataset_name, model="ResNet"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset_name == 'wikiart':
        df_path = 'DATA/Dataset/wikiart/wikiartINFL.pkl'
        dataset_path = '/home/tliberatore2/Reproduction-of-ArtSAGENet/wikiart/'
        output_path = 'DATA/Dataset/wikiart/wikiartINFL_clip.pkl'
    elif dataset_name == 'idesigner':
        df_path = 'DATA/Dataset/iDesigner/idesigner_influences_cropped.pkl'
        dataset_path = 'DATA/Dataset/iDesigner/designer_image_train_v2_cropped/'
        output_path = 'DATA/Dataset/iDesigner/idesignerINFL.pkl'

    df = pd.read_pickle(df_path)
    df = df.drop(columns=['image_features', 'image_text_features', 'text_features', 'additional_styles'])

    if model == "clip":
        clip_process_data(dataset_name, dataset_path, df, output_path,device)
    elif model == "ResNet":
        preprocess_data(df,dataset_path,output_path,dataset_name, device)





if __name__ == '__main__':
    start_time = time.time() 
    parser = argparse.ArgumentParser(description="Create dataset for triplet loss network on wikiart to predict influence.")
    parser.add_argument('--dataset_name', type=str, default='wikiart', choices=['wikiart', 'fashion'])
    parser.add_argument('--model', type=str, default='ResNet', choices=['ResNet', 'clip'])
    args = parser.parse_args()
    main(args.dataset_name, args.model)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required to extract the features: {:.2f} seconds".format(elapsed_time))

