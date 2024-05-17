import pandas as pd
from datasets import Dataset
from renumics import spotlight

dataset_name = "wikiart"
feature = 'image_text_features'
if dataset_name == 'wikiart':
    df = pd.read_pickle('DATA/Dataset/wikiart/wikiart_full_combined_no_artist_filtered.pkl')
elif dataset_name == 'fashion':
    df = pd.read_pickle('DATA/Dataset/iDesigner/idesigner_influences_cropped_features.pkl')
# all_artist_names = set(df['artist_name'])
# df['influenced_by'] = df['influenced_by'].apply(lambda artists_list: [artist for artist in artists_list if artist in all_artist_names])

df["image_features"] = df["image_features"].apply(lambda x: x.numpy())
df['influenced_by'] = df['influenced_by'].apply(lambda x: ' '.join(x))
if dataset_name == 'wikiart':
    df.drop(columns = ['title'],inplace=True)
    df.text_features = df.text_features.apply(lambda x: x.numpy())
    df.image_text_features = df.image_text_features.apply(lambda x: x.numpy())

import torch
import torch.nn as nn

class TripletResNet_features(nn.Module):
    def __init__(self, input_size):
        super(TripletResNet_features, self).__init__()

        hidden_size_1 = input_size//2
        hidden_size_2 = input_size//4
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size_1, hidden_size_2),
             nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size_2,input_size )
        )


    def forward_once(self, x):
        # Pass input through the ResNet
        output = self.model(x)
        return output

    def forward(self, anchor, positive, negative):
        # Forward pass for both images
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)
        return anchor_output, positive_output, negative_output
    
model_path = f'trained_models/{dataset_name}/ResNet34/TripletResNet_{feature}_posfaiss_negrandom_100_margin1/model.pth'
model = TripletResNet_features(df.loc[0,feature].shape[0])
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
df[feature] = df[feature].apply(lambda x: torch.from_numpy(x))
df[f'trained_{feature}'] = df[feature].apply(lambda x: model.forward_once(x).detach())
df[f'trained_{feature}'] = df[f'trained_{feature}'].apply(lambda x: x.numpy())
df[feature] = df[feature].apply(lambda x: x.numpy())


dataset = Dataset.from_pandas(df)

spotlight.show(dataset, dtype={f'trained_{feature}':spotlight.Embedding})