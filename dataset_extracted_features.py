import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
from triplet_network import clean
import numpy as np

import torch
from torchvision import models, transforms
from PIL import Image

import torch
from torchvision import models, transforms
from PIL import Image
from transformers import GPT2Tokenizer, GPT2Model

import ast

# Check if CUDA is available, otherwise use CPU


# Load pre-trained ResNet-152



# Function to extract features from an image
def image_features(image_path,device):
    weights = models.ResNet34_Weights.DEFAULT
    resnet152 = models.resnet34(weights=weights)

    # Move model to the selected device
    resnet152 = resnet152.to(device)

    # Remove the final fully connected layer
    resnet152 = torch.nn.Sequential(*(list(resnet152.children())[:-1]))

    # Set model to evaluation mode
    resnet152.eval()

    # Define image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        #transforms.CenterCrop(224),
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
        features = resnet152(image_tensor)
    # Flatten the features
    features = features.squeeze().cpu().numpy() 
    return features





# Define function to get vector representation of a sentence
def get_sentence_vector(sentence,device):
    # Tokenize input text
    device = 'mps' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2').to(device)

    input_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt').to(device)
    # Get model output
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state
    # Average the last hidden states to get sentence representation
    sentence_vector = torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()
    print(sentence_vector.shape, sentence_vector.dtype)
    # Convert to numpy array
    return sentence_vector

def get_embedding(text, model, tokenizer, device):
    # Tokenize the text
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(device)

    # Get model output
    with torch.no_grad():
        outputs = model(input_ids)
        embedding = outputs.last_hidden_state.squeeze().mean(dim=0).cpu().numpy()

    return embedding

def get_text_features(df,device):
    # Load GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_basic_tokenize=False, do_lower_case=False)
    model = GPT2Model.from_pretrained('gpt2').to(device)


    # Process each column separately and get embeddings
    embeddings = []
    for col in ['style_classification','tags', 'artist_name','artist_school','timeframe_estimation','title']:#'artist_name', 'artist_school', 'style_classification',
        column_embeddings = []
        for text in df[col]:
            embedding = get_embedding(text, model, tokenizer,device)
            column_embeddings.append(embedding)
        embeddings.append(column_embeddings)

    # Average the embeddings
    average_embeddings = [sum(x) / len(x) for x in zip(*embeddings)]
    df['text_features'] = average_embeddings
    return df

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def pca_transform(df,feature,n_components=3):
    # Standardize the data
    features = df[feature].tolist()
    features = [np.array(x) for x in features]
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(features)
    df['pca_features'] = principalComponents.tolist()
    #principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    return df

def preprocess_data(df,device):
    df['tags'] = df['tags'].fillna(' ')
    df['title'] = df.apply(lambda x: x.relative_path.split('/')[-1].split('.')[0].split('_')[1], axis=1)
    df = get_text_features(df,device)
    if 'extracted_features' not in df.columns:

        df['image_features'] = df.apply(lambda x: image_features(x.relative_path,device), axis=1)
    else:
        df['image_features'] = df['extracted_features']
        df.drop(columns=['extracted_features'], inplace=True)
    df.text_features = df.text_features.apply(lambda x: torch.tensor(x))
    df.image_features = df.image_features.apply(lambda x: ast.literal_eval(x))
    df.image_features = df.image_features.apply(lambda x: torch.tensor(x))
    df['image_text_features'] = df.apply(lambda x: torch.cat([x['image_features'], x['text_features']]), axis=1)
    df['image_text_features'] = df['image_text_features'].apply(lambda x: x.tolist())
    df['image_features'] = df['image_features'].apply(lambda x: x.tolist())
    df['text_features'] = df['text_features'].apply(lambda x: x.tolist())
    df = pca_transform(df,'image_text_features',n_components=3)
    df.to_csv('DATA/Dataset/wikiart_1000_combined.csv', index=False)
    return df

def main():
    device = 'mps' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # df = pd.read_csv('DATA/Dataset/wikiart_full_influence_filtered.csv')
    # df = clean(df)
    df = pd.read_csv('DATA/Dataset/wikiart_full_features_34.csv')
    df = df[:1000]
    preprocess_data(df,device)


    


    # df['extracted_features'] = df['relative_path'].apply(lambda x: extract_features(x,device).tolist())
    # df.to_csv('DATA/Dataset/wikiart_full_resnet34.csv', index=False)
if __name__ == '__main__':
    main()

