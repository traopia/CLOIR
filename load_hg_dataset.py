import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from datasets import Dataset, Array2D, Features, Value

class CustomDataset(Dataset):
    def __init__(self, image_dir, metadata_file, transform=None):
        self.image_dir = image_dir
        self.metadata = pd.read_csv(metadata_file)
        self.metadata['date'] = self.metadata['date'].astype(str)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.metadata.iloc[idx, 1])
        image = Image.open(img_name)
        metadata = self.metadata.iloc[idx, ]  
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'metadata': metadata}

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


# Example usage
image_dir = '/home/tliberatore2/Reproduction-of-ArtSAGENet/wikiart/'
metadata_file = 'DATA/Dataset/wikiart_full_influence_filtered.csv'
transform = None  # You can define your image transformations here

custom_dataset = CustomDataset(image_dir=image_dir, metadata_file=metadata_file, transform=transform)

metadata_features = {
    'relative_path': Value('string'),
    'style_classification': Value('string'),
    'artist_attribution': Value('string'),
    'timeframe_estimation': Value('string'),
    'tag_prediction': Value('string'),
    'mode': Value('string'),
    'date': Value('string'),
    'artist_name': Value('string'),
    'additional_styles': Value('string'),
    'artist_school': Value('string'),
    'tags': Value('string'),
    'influenced_by': Value('string')
}

# Combine image and metadata features
features = Features({
    'image': Array2D(dtype='uint8', shape=(None, None, 3)),  # Adjust shape and dtype based on your image format
    'metadata': metadata_features
})

data_dict = {
    'image': [],
    'metadata': []
}

for item in custom_dataset:
    data_dict['image'].append(item['image'])  # Convert image to numpy array
    data_dict['metadata'].append(item['metadata'])

# Step 2: Convert dictionary to Hugging Face Dataset
hf_dataset = Dataset.from_dict(data_dict, features=features)

# Step 3: Save dataset to disk
hf_dataset.save_to_disk('wikiart_influence')
