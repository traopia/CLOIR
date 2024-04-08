# from datasets import load_dataset

# # d = load_dataset('/home/tliberatore2/Reproduction-of-ArtSAGENet/wikiart/')
# # #dataset = load_dataset("imagefolder", data_dir="/home/tliberatore2/Reproduction-of-ArtSAGENet/wikiart/")
# # d.save_to_disk('wikiart_hg')




# import os
# import pandas as pd
# from PIL import Image
# from torchvision import transforms
# from tqdm import tqdm
# from datasets import Dataset
# import torch

# # Step 1: Create a pandas DataFrame with relative paths to images
# def create_dataframe(data_dir):
#     image_files = []
#     for root, dirs, files in os.walk(data_dir):
#         for file in files:
#             if file.endswith(('.jpg', '.jpeg', '.png')):
#                 image_files.append(os.path.join(root, file))
#     df = pd.DataFrame({'image_path': image_files})
#     return df

# # Step 2: Custom dataset class to load images
# class ImageDataset(torch.utils.data.Dataset):
#     def __init__(self, df, transform=None):
#         self.df = df[:10]
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         img_path = self.df.iloc[idx]['image_path']
#         img = Image.open(img_path).convert('RGB')
#         if self.transform:
#             img = self.transform(img)
#         return {'image':img, 'image_path': img_path}

# # Step 3: Register the dataset with Hugging Face's datasets library
# def register_hf_dataset(data_dir, dataset_name):
#     df = create_dataframe(data_dir)
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # adjust as per your requirements
#         #transforms.ToTensor(),
#     ])
#     dataset = ImageDataset(df, transform=transform)
#     hf_dataset = Dataset.from_pandas(df)
    
#     hf_dataset = hf_dataset.map(lambda x: {'image': Image.open(x['image_path']).convert('RGB')})

#     hf_dataset.save_to_disk(f'{dataset_name}')
#     return dataset, hf_dataset

# # Example usage:
# data_dir = '/home/tliberatore2/Reproduction-of-ArtSAGENet/wikiart/'
# dataset_name = 'wikiart_hg'
# dataset , hf= register_hf_dataset(data_dir, dataset_name)

import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from datasets import Dataset
import torch

# Step 1: Create a pandas DataFrame with relative paths to images
def create_dataframe(data_dir):
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    df = pd.DataFrame({'image_path': image_files})
    return df

# Step 2: Custom dataset class to load images
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df_subset, transform=None):
        self.df = df_subset
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'image_path': img_path}

# Step 3: Register the dataset with Hugging Face's datasets library
def register_hf_dataset(data_dir, dataset_name, batch_size=100):
    df = create_dataframe(data_dir)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # adjust as per your requirements
    ])

    # Split dataframe into smaller subsets or batches
    num_batches = len(df) // batch_size
    if len(df) % batch_size != 0:
        num_batches += 1

    datasets = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        df_subset = df[start_idx:end_idx]
        
        dataset = ImageDataset(df_subset, transform=transform)
        datasets.append(dataset)

    hf_datasets = []
    for i, dataset in enumerate(datasets):
        hf_dataset = Dataset.from_pandas(datasets[i].df)
        hf_dataset = hf_dataset.map(lambda x: {'image': Image.open(x['image_path']).convert('RGB')})
        hf_dataset.save_to_disk(f'{dataset_name}_{i}')
        hf_datasets.append(hf_dataset)

    return datasets, hf_datasets

# Example usage:
data_dir = '/home/tliberatore2/Reproduction-of-ArtSAGENet/wikiart/'
dataset_name = 'wikiart_hg'
datasets, hf_datasets = register_hf_dataset(data_dir, dataset_name)

