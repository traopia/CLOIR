import clip_test
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle

class WikiartDataset(Dataset):
    def __init__(self, df, image_dir, preprocess):
        self.df = df
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.context_length = 77

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir + row['relative_path']
        image = self.preprocess(Image.open(image_path))

        title = row['relative_path'].split('/')[-1].split('.')[0].split('_')
        if len(title) == 2:
            title = title[1]
        #tags = row['tags'] if pd.notna(row['tags']) else ' '
        text = f"{row['style_classification']} {row['timeframe_estimation']} {row['artist_school']} {title}"
        
        # Truncate text to fit within the context length limit
        # text_tokens = clip.tokenize([text])[0]
        # if len(text_tokens) > self.context_length:
        #     text_tokens = text_tokens[:self.context_length]
        #     text = clip.tokenize.decode(text_tokens.tolist())
        
        return image, text

def extract_features(model, dataloader, df, device):
    with torch.no_grad():
        with open('WikiartCLIP.pkl', 'wb') as file:
            for images, texts in dataloader:
                images = images.to(device)
                texts = clip_test.tokenize(texts).to(device)

                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                df['clip_image_features'] = list(image_features.cpu())
                df['clip_text_features'] = list(text_features.cpu())
                pickle.dump(df, file)
                #del images, texts  # Free up memory

    return df

def extract_features(model, dataloader, df, device):
    image_features_list = []
    text_features_list = []
    
    with torch.no_grad():
        for images, texts in dataloader:
            images = images.to(device)
            texts = clip_test.tokenize(texts).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            image_features_list.append(image_features.cpu())
            text_features_list.append(text_features.cpu())

    df['clip_image_features'] = image_features_list
    df['clip_text_features'] = text_features_list

    return df


def clip_process_data(dataset_name, dataset_path, df, output_path, device):
    batch_size = 16
    model, preprocess = clip_test.load("ViT-B/32", device=device)

    dataset = WikiartDataset(df, dataset_path, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    df = df.drop(columns=['concatenated_text', 'tag_prediction', 'title'], errors='ignore')

    df = extract_features(model, dataloader,df, device)
    df.to_pickle(output_path)

    return df


def extract_features(model, dataloader, device):
    image_features_list = []
    text_features_list = []

    with torch.no_grad():
        for images, texts in dataloader:
            images = images.to(device)
            texts = clip_test.tokenize(texts).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            image_features_list.append(image_features.cpu())
            text_features_list.append(text_features.cpu())

    return torch.cat(image_features_list), torch.cat(text_features_list)



def clip_process_data(dataset_name, general_path, df, dataset_outpath, device):
    batch_size = 16
    model, preprocess = clip_test.load("ViT-B/32", device=device)

    dataset = WikiartDataset(df, general_path, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    image_features, text_features = extract_features(model, dataloader, device)


    df['clip_image_features'] = list(image_features)
    if dataset_name =="wikiart":
        df['clip_text_features'] = list(text_features)
    df = df.drop(columns=['concatenated_text', 'tag_prediction', 'title'])
    df.to_pickle(dataset_outpath)

    return df

def main(dataset_name, model="ResNet"):
    device = "cuda" if torch.cuda.is_available() else "mps"
    if dataset_name == 'wikiart':
        df_path = 'DATA/Dataset/wikiart/wikiartINFL.pkl'
        dataset_path = 'wikiart/'
        output_path = 'DATA/Dataset/wikiart/wikiartINFL_clip_1.pkl'
    elif dataset_name == 'idesigner':
        df_path = 'DATA/Dataset/iDesigner/idesigner_influences_cropped.pkl'
        dataset_path = 'DATA/Dataset/iDesigner/designer_image_train_v2_cropped/'
        output_path = 'DATA/Dataset/iDesigner/idesignerINFL.pkl'

    df = pd.read_pickle(df_path)
    df = df.drop(columns=['image_features', 'image_text_features', 'text_features', 'additional_styles'], errors='ignore')

    if model == "clip":
        clip_process_data(dataset_name, dataset_path, df, output_path, device)


main("wikiart", "clip")