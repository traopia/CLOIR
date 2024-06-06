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



def main(dataset_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    if dataset_name == 'wikiart':
        df_path = 'DATA/Dataset/wikiart_full_influence.pkl'
        dataset_path = '/home/tliberatore2/Reproduction-of-ArtSAGENet/wikiart/' #'wikiart/'
        output_path = 'DATA/Dataset/wikiartINFL.pkl'

    elif dataset_name == 'idesigner':
        df_path = 'DATA/Dataset/iDesigner/idesigner_influences_cropped.pkl'
        dataset_path = 'DATA/Dataset/iDesigner/designer_image_train_v2_cropped/'
        output_path = 'DATA/Dataset/iDesigner/idesignerINFL.pkl'

    df = pd.read_pickle(df_path)
    df = clean(df)
    preprocess_data(df,dataset_path,output_path,dataset_name, device)

if __name__ == '__main__':
    start_time = time.time() 
    dataset_name = 'idesigner'
    main(dataset_name)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required to extract the features: {:.2f} seconds".format(elapsed_time))

