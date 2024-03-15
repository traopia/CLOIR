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



def clean(df):
    df.dropna(subset=['influenced_by'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    unique_values = df['artist_name'].explode().unique()
    df['influenced_by'] = df['influenced_by'].apply(lambda x: x.split(', '))
    df['influenced_by'] = df['influenced_by'].apply(lambda x: [i for i in x if i in unique_values])
    df = df[df['influenced_by'].map(len) > 0]
    df.reset_index(drop=True, inplace=True)
    return df


def image_features(image_path,device):

    weights = models.ResNet34_Weights.DEFAULT
    resnet34 = models.resnet34(weights=weights)
    resnet34 = resnet34.to(device)
    resnet34 = torch.nn.Sequential(*(list(resnet34.children())[:-1]))

    # Set model to evaluation mode
    resnet34.eval()
    # Define image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_path = 'wikiart/' + image_path
    image = Image.open(full_path)
    # Preprocess the image
    image_tensor = preprocess(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    # Move tensor to the selected device
    image_tensor = image_tensor.to(device)
    # Extract features
    with torch.no_grad():
        features = resnet34(image_tensor)
    # Flatten the features
    features = features.squeeze().cpu()

    return features

def get_image_features(df,device):
    embeddings = []
    for image_path in df['relative_path']:
        embedding = image_features(image_path,device)
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
    df.loc[:, 'title']  = df.apply(lambda x: x.relative_path.split('/')[-1].split('.')[0].split('_')[1], axis=1)
    df.loc[:, 'tags'] = df['tags'].fillna(' ')
    # Concatenate text from different columns
    df.loc[:,'concatenated_text'] = df['artist_attribution'].astype(str) + ' ' + df['style_classification'].astype(str)  + ' '  + df['title'].astype(str) + ' ' + df['tags'].astype(str) + ' ' + df['timeframe_estimation'].astype(str) + ' ' + df['artist_school'].astype(str) 

    # Get embeddings for the concatenated text
    embeddings = []
    for text in df['concatenated_text']:
        embedding = get_embedding(text, model, tokenizer, device)
        embeddings.append(embedding)

    return embeddings



def preprocess_data(df,device):
    '''Preprocess the data and save it as a pickle file'''

    df['text_features'] = get_text_features(df,device)
    df['image_features'] = get_image_features(df,device)
    df['image_text_features'] = df.apply(lambda x: torch.cat([x['image_features'], x['text_features']]), axis=1)

    df.to_pickle('DATA/Dataset/wikiart_full_combined_try.pkl')

    return df


def get_dictionaries(df):
    dict_influenced_by = df.groupby('artist_name')['influenced_by'].first().to_dict()
    artist_to_paintings = {}
    for index, row in df.iterrows():
        artist = row['artist_name']
        artist_to_paintings.setdefault(artist, []).append(index)
    artist_to_influencer_paintings = {artist: [painting for influencer in influencers if influencer in artist_to_paintings for painting in artist_to_paintings[influencer]] for artist, influencers in dict_influenced_by.items()}
    if os.path.exists('DATA/dict') == False:
        os.makedirs('DATA/dict')
    file_path = 'DATA/dict/artist_to_influencer_paintings.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(artist_to_influencer_paintings, file)
    return artist_to_influencer_paintings

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    df = pd.read_pickle('DATA/Dataset/wikiart_full_influence_filtered.csv')
    df = clean(df)
    get_dictionaries(df)
    preprocess_data(df,device)

if __name__ == '__main__':
    start_time = time.time() 
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required to extract the features: {:.2f} seconds".format(elapsed_time))

