import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from create_data_loader import TripletLossDataset_features
import time 
import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random
random.seed(42)

def accuracy_triplet(anchor_output, positive_output, negative_output):
    distance_positive = F.pairwise_distance(anchor_output, positive_output)
    distance_negative = F.pairwise_distance(anchor_output, negative_output)
    return torch.mean((distance_positive < distance_negative).float()).cpu().numpy()


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
    


def train_artist(model, epochs, train_loader, criterion, optimizer, device, name_model, feature_extractor_name):
    model.train()
    running_loss_train = 0.0
    running_accuracy_train = 0.0
    loss_plot_train = []
    accuracy_plot_train = []

    for epoch in range(epochs):
        for anchor_batch, positive_batch, negative_batch in train_loader:
            anchor_batch, positive_batch, negative_batch = anchor_batch.to(device), positive_batch.to(device), negative_batch.to(device)
            optimizer.zero_grad()
            output1, output2, output3 = model(anchor_batch, positive_batch, negative_batch)
            loss = criterion(output1, output2, output3)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()
            accuracy = accuracy_triplet(output1, output2, output3)
            running_accuracy_train += accuracy

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {running_loss_train / len(train_loader):.4f}, Train Accuracy: {running_accuracy_train/ len(train_loader)}%")

        wandb.log({"Train Loss": running_loss_train / len(train_loader), "Train Accuracy": running_accuracy_train/ len(train_loader)})
        loss_plot_train.append(running_loss_train / len(train_loader))
        accuracy_plot_train.append(running_accuracy_train / len(train_loader))
        running_loss_train = 0.0
        running_accuracy_train = 0.0

    if not os.path.exists(f'trained_models/Artists/{feature_extractor_name}_{name_model}'):
        os.makedirs(f'trained_models/Artists/{feature_extractor_name}_{name_model}')
    torch.save(model.state_dict(), f'trained_models/Artists/{feature_extractor_name}_{name_model}/model.pth')
    metrics = {'loss_plot_train': loss_plot_train, 'accuracy_plot_train': accuracy_plot_train}
    torch.save(metrics, f'trained_models/Artists/{feature_extractor_name}_{name_model}/metrics.pth')

    plt.plot(loss_plot_train, label='Training Loss')
    plt.plot(accuracy_plot_train, label='Training Accuracy')
    plt.title('Model Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.savefig(f'trained_models/Artists/{feature_extractor_name}_{name_model}/loss_plot.png')




def train(model,epochs, train_loader, val_loader, criterion, optimizer, device,name_model,feature_extractor_name):
    model.train()
    running_loss_train, running_loss_val = 0.0, 0.0
    running_accuracy_train, running_accuracy_val = 0.0, 0.0
    loss_plot_train, loss_plot_val = [], []
    accuracy_plot_train, accuracy_plot_val = [], []

    for epoch in range(epochs):
        for anchor_batch, positive_batch, negative_batch in train_loader:
            anchor_batch, positive_batch, negative_batch = anchor_batch.to(device), positive_batch.to(device), negative_batch.to(device)
            optimizer.zero_grad()
            output1, output2, output3 = model(anchor_batch, positive_batch, negative_batch)
            loss = criterion(output1, output2, output3)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()
            accuracy = accuracy_triplet(output1, output2, output3)
            running_accuracy_train += accuracy

        print(f"Epoch [{epoch + 1}/{10}], Train Loss: {running_loss_train / len(train_loader):.4f}, Train Accuracy: {running_accuracy_train/ len(train_loader)}%")

        model.eval()
        for anchor_batch, positive_batch, negative_batch in val_loader:
            anchor_batch, positive_batch, negative_batch = anchor_batch.to(device), positive_batch.to(device), negative_batch.to(device)
            output1, output2, output3 = model(anchor_batch, positive_batch, negative_batch)
            loss = criterion(output1, output2, output3)
            running_loss_val += loss.item()
            accuracy = accuracy_triplet(output1, output2, output3)
            running_accuracy_val += accuracy

        print(f"Epoch [{epoch + 1}/{10}], Val Loss: {running_loss_val / len(val_loader):.4f}, Val Accuracy: {running_accuracy_val/ len(val_loader)}")

        wandb.log({"Train Loss": running_loss_train / len(train_loader), "Val Loss": running_loss_val / len(val_loader), "Train Accuracy": running_accuracy_train/ len(train_loader), "Val Accuracy": running_accuracy_val/ len(val_loader)})
        loss_plot_train.append(running_loss_train / len(train_loader))
        loss_plot_val.append(running_loss_val / len(val_loader))

        accuracy_plot_train.append(running_accuracy_train / len(train_loader))
        accuracy_plot_val.append(running_accuracy_val / len(val_loader))
        running_loss_train, running_loss_val = 0.0, 0.0
        running_accuracy_train, running_accuracy_val = 0.0, 0.0
 

    if os.path.exists(f'trained_models/{feature_extractor_name}/{name_model}') == False:
        os.makedirs(f'trained_models/{feature_extractor_name}/{name_model}')
    torch.save(model.state_dict(), f'trained_models/{feature_extractor_name}/{name_model}/model.pth')
    metrics = {'loss_plot_train': loss_plot_train, 'loss_plot_val': loss_plot_val, 'accuracy_plot_train': accuracy_plot_train, 'accuracy_plot_val': accuracy_plot_val}
    torch.save(metrics, f'trained_models/{feature_extractor_name}/{name_model}/metrics.pth')

    plt.plot(loss_plot_train, label='Training Loss')
    plt.plot(loss_plot_val, label='Validation Loss')
    plt.plot(accuracy_plot_train, label='Training Accuracy')
    plt.plot(accuracy_plot_val, label='Validation Accuracy')
    plt.title('Model Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.savefig(f'trained_models/{feature_extractor_name}/{name_model}/loss_plot.png')
    

def main(feature,feature_extractor_name, num_examples,positive_based_on_similarity, negative_based_on_similarity):    
    epochs = wandb.config.epochs
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    margin = wandb.config.margin
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    how_feature_positive = 'posfaiss' if positive_based_on_similarity else 'posrandom'
    how_feature_negative = 'negfaiss' if negative_based_on_similarity else 'negrandom'
    dataset_train = torch.load(f'DATA/Dataset_toload/{feature_extractor_name}/train_dataset_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}.pt')
    dataset_val = torch.load(f'DATA/Dataset_toload/{feature_extractor_name}/val_dataset_{feature}_{how_feature_positive}_{how_feature_negative}_10.pt')

    augmentations = [
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(degrees=10),  # Randomly rotate images by a maximum of 10 degrees
    transforms.CenterCrop(size=224),  # Randomly crop images to size 224x224
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # Randomly adjust brightness, contrast, saturation, and hue
    ]

    # Define a composite transformation that randomly applies augmentations
    transform = transforms.Compose([
        transforms.RandomApply(augmentations, p=0.5),  # Randomly apply augmentations with a probability of 0.5
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the tensor values to range [-1, 1]
    ])    
    dataset_train.transform = transform
    dataset_val.transform = transform
    tripleloss_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    tripleloss_loader_val = DataLoader(dataset_val, shuffle=False, batch_size=batch_size)
    net = TripletResNet_features(dataset_train.dimension).to(device)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.Adam(net.parameters(), lr =  lr)
    train(net,epochs, tripleloss_loader_train, tripleloss_loader_val, criterion, optimizer, device, f'TripletResNet_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}_margin{margin}',feature_extractor_name)

def main_artist(feature,artist_name, num_examples,positive_based_on_similarity, negative_based_on_similarity):
    epochs = wandb.config.epochs
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    margin = wandb.config.margin
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    how_feature_positive = 'posfaiss' if positive_based_on_similarity else 'posrandom'
    how_feature_negative = 'negfaiss' if negative_based_on_similarity else 'negrandom'
    dataset_train = torch.load(f'DATA/Dataset_toload/Artists/{artist_name}_train_dataset_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}.pt')

    augmentations = [
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(degrees=10),  # Randomly rotate images by a maximum of 10 degrees
    transforms.CenterCrop(size=224),  # Randomly crop images to size 224x224
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # Randomly adjust brightness, contrast, saturation, and hue
    ]

    # Define a composite transformation that randomly applies augmentations
    transform = transforms.Compose([
        transforms.RandomApply(augmentations, p=0.5),  # Randomly apply augmentations with a probability of 0.5
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the tensor values to range [-1, 1]
    ])    
    dataset_train.transform = transform
    tripleloss_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    net = TripletResNet_features(dataset_train.dimension).to(device)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.Adam(net.parameters(), lr =  lr)
    train_artist(net,epochs, tripleloss_loader_train, criterion, optimizer, device, f'TripletResNet_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}_margin{margin}',artist_name)

if __name__ == "__main__":
    start_time = time.time() 

    parser = argparse.ArgumentParser(description="Train Triplet Loss Contrastive Network on Wikiart to predict influence.")
    parser.add_argument('--feature', type=str, default='image_features', help='image_features text_features image_text_features')
    parser.add_argument('--artist_splits', action='store_true',help= 'create dataset excluding a gievn artist from training set' )
    parser.add_argument('--feature_extractor_name', type=str, default = 'ResNet34', help= ['ResNet34', 'ResNet34_newsplit' 'ResNet152'])
    parser.add_argument('--num_examples', type=int, default=10, help= 'How many examples for each anchor')
    parser.add_argument('--positive_based_on_similarity',action='store_true',help='Sample positive examples based on vector similarity or randomly')
    parser.add_argument('--negative_based_on_similarity', action='store_true',help='Sample negative examples based on vector similarity or randomly')
    args = parser.parse_args()
    wandb.init(
    # set the wandb project where this run will be logged
    project="Triplet_Network_Wikiart_predict_Influence",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0005,
    "architecture": "Triplet Network",
    "dataset": "Wikiart",
    "batch_size": 32,
    "epochs": 10,
    "margin": 2, 
    "num_examples": args.num_examples,
    "feature": args.feature,
    "positive_based_on_similarity": args.positive_based_on_similarity,
    "negative_based_on_similarity": args.negative_based_on_similarity,
    "feature_extractor_name": 'pablo-picasso'#args.feature_extractor_name
    }

    )
    if args.artist_splits:
        main_artist(args.feature,args.feature_extractor_name, args.num_examples,args.positive_based_on_similarity, args.negative_based_on_similarity)
    else:
        main(args.feature,args.feature_extractor_name, args.num_examples,args.positive_based_on_similarity, args.negative_based_on_similarity)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required for training : {:.2f} seconds".format(elapsed_time))
    wandb.finish()