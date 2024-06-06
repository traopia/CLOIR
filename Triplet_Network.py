import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ast
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
from torch.utils.data import DataLoader, Dataset
from create_data_loader import TripletLossDataset_features
import time 
import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd

import random
random.seed(42)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class TripletResNet_features(nn.Module):
    def __init__(self, input_size):
        super(TripletResNet_features, self).__init__()

        hidden_size_1 = input_size//2
        hidden_size_2 = hidden_size_1//2
        output_size =  hidden_size_2//2 #128 
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size_2,output_size )
        )


    def forward_once(self, x):
        output = self.model(x)
        return output

    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)
        return anchor_output, positive_output, negative_output
    




def train(model,epochs, train_loader, val_loader, criterion, optimizer, device,trained_model_path):
    model.train()
    running_loss_train, running_loss_val = 0.0, 0.0
    loss_plot_train, loss_plot_val = [], []


    for epoch in range(epochs):
        for anchor_batch, positive_batch, negative_batch in train_loader:
            anchor_batch, positive_batch, negative_batch = anchor_batch.to(device), positive_batch.to(device), negative_batch.to(device)
            optimizer.zero_grad()
            output1, output2, output3 = model(anchor_batch, positive_batch, negative_batch)
            # batch_size, n_samples, embedding_dim = output1.size()
            # output1 = output1.view(-1, embedding_dim)
            # output2 = output2.view(-1, embedding_dim)
            # output3 = output3.view(-1, embedding_dim)

            loss = criterion(output1, output2, output3)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()


        model.eval()
        for anchor_batch, positive_batch, negative_batch in val_loader:
            anchor_batch, positive_batch, negative_batch = anchor_batch.to(device), positive_batch.to(device), negative_batch.to(device)
            output1, output2, output3 = model(anchor_batch, positive_batch, negative_batch)
            loss = criterion(output1, output2, output3)
            running_loss_val += loss.item()


        print(f"Epoch [{epoch + 1}/{10}],  Train Loss: {running_loss_train / len(train_loader):.4f},Val Loss: {running_loss_val / len(val_loader):.4f}")

        wandb.log({"Train Loss": running_loss_train / len(train_loader), "Val Loss": running_loss_val / len(val_loader)})
        loss_plot_train.append(running_loss_train / len(train_loader))
        loss_plot_val.append(running_loss_val / len(val_loader))


        running_loss_train, running_loss_val = 0.0, 0.0



    if os.path.exists(trained_model_path) == False:
        os.makedirs(trained_model_path)
    torch.save(model.state_dict(), f'{trained_model_path}/model.pth')
    metrics = {'loss_plot_train': loss_plot_train, 'loss_plot_val': loss_plot_val}
    torch.save(metrics, f'{trained_model_path}/metrics.pth')

    plt.plot(loss_plot_train, label='Training Loss')
    plt.plot(loss_plot_val, label='Validation Loss')

    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss ')
    plt.legend()
    plt.savefig(f'{trained_model_path}/loss_plot.png')
    

def train(model, epochs, train_loader, val_loader, criterion, optimizer, device, trained_model_path):
    if os.path.exists(trained_model_path) == False:
        os.makedirs(trained_model_path)
    model.train()
    loss_plot_train, loss_plot_val = [], []
    best_val_loss = float('inf')
    patience = 10 # Number of epochs to wait for improvement before early stopping
    early_stopping_counter = 0

    for epoch in range(epochs):
        running_loss_train, running_loss_val = 0.0, 0.0
        
        # Training phase
        for anchor_batch, positive_batch, negative_batch in train_loader:
            anchor_batch, positive_batch, negative_batch = anchor_batch.to(device), positive_batch.to(device), negative_batch.to(device)
            optimizer.zero_grad()
            output1, output2, output3 = model(anchor_batch, positive_batch, negative_batch)
            loss = criterion(output1, output2, output3)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            for anchor_batch, positive_batch, negative_batch in val_loader:
                anchor_batch, positive_batch, negative_batch = anchor_batch.to(device), positive_batch.to(device), negative_batch.to(device)
                output1, output2, output3 = model(anchor_batch, positive_batch, negative_batch)
                loss = criterion(output1, output2, output3)
                running_loss_val += loss.item()

        # Compute average losses
        avg_loss_train = running_loss_train / len(train_loader)
        avg_loss_val = running_loss_val / len(val_loader)

        # Log and print losses
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss_train:.4f}, Val Loss: {avg_loss_val:.4f}")

        # Check for early stopping
        if avg_loss_val < best_val_loss:
            best_val_loss = avg_loss_val
            early_stopping_counter = 0
            # Save the model if validation loss has decreased
            torch.save(model.state_dict(), f'{trained_model_path}/model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"No improvement in validation loss for {patience} epochs. Early stopping.")
                break

        # Log losses for visualization
        loss_plot_train.append(avg_loss_train)
        loss_plot_val.append(avg_loss_val)

    # Save training metrics and plots
    save_metrics_plots(loss_plot_train, loss_plot_val, trained_model_path)


def save_metrics_plots(loss_plot_train, loss_plot_val, trained_model_path):
    # Save metrics and plots
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)
    torch.save({'loss_plot_train': loss_plot_train, 'loss_plot_val': loss_plot_val}, f'{trained_model_path}/metrics.pth')

    # Plot and save the training and validation loss
    plt.plot(loss_plot_train, label='Training Loss')
    plt.plot(loss_plot_val, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{trained_model_path}/loss_plot.png')





def main(dataset_name, feature,data_split, num_examples,positive_based_on_similarity, negative_based_on_similarity):    
    epochs = wandb.config.epochs
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    margin = wandb.config.margin
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    how_feature_positive = 'posfaiss' if positive_based_on_similarity else 'posrandom'
    how_feature_negative = 'negfaiss' if negative_based_on_similarity else 'negrandom'
    dataset_train = torch.load(f'DATA/Dataset_toload/{dataset_name}/{data_split}/train_dataset_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}.pt')
    dataset_val = torch.load(f'DATA/Dataset_toload/{dataset_name}/{data_split}/val_dataset_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}.pt')
    tripleloss_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    tripleloss_loader_val = DataLoader(dataset_val, shuffle=False, batch_size=batch_size)
    net = TripletResNet_features(dataset_train.dimension).to(device)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    trained_model_path = f'trained_models/{dataset_name}/{data_split}/TripletResNet_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}_margin{margin}_notrans_epoch_{epochs}'
    train(net,epochs, tripleloss_loader_train, tripleloss_loader_val, criterion, optimizer, device, trained_model_path)



def train_artist(model, epochs, train_loader, criterion, optimizer, device, trained_model_path):
    model.train()
    running_loss_train = 0.0
    loss_plot_train = []

    for epoch in range(epochs):
        for anchor_batch, positive_batch, negative_batch in train_loader:
            anchor_batch, positive_batch, negative_batch = anchor_batch.to(device), positive_batch.to(device), negative_batch.to(device)
            optimizer.zero_grad()
            output1, output2, output3 = model(anchor_batch, positive_batch, negative_batch)
            loss = criterion(output1, output2, output3)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()


        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {running_loss_train / len(train_loader):.4f}%")

        wandb.log({"Train Loss": running_loss_train / len(train_loader)})
        loss_plot_train.append(running_loss_train / len(train_loader))

        running_loss_train = 0.0


    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)
    torch.save(model.state_dict(), f'{trained_model_path}/model.pth')
    metrics = {'loss_plot_train': loss_plot_train}
    torch.save(metrics, f'{trained_model_path}/metrics.pth')

    plt.plot(loss_plot_train, label='Training Loss')

    plt.title('Model Loss ')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{trained_model_path}/loss_plot.png')

def main_artist(dataset_name, feature,data_split, num_examples,positive_based_on_similarity, negative_based_on_similarity):
    epochs = 10#wandb.config.epochs
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    margin = wandb.config.margin
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    how_feature_positive = 'posfaiss' if positive_based_on_similarity else 'posrandom'
    how_feature_negative = 'negfaiss' if negative_based_on_similarity else 'negrandom'
    if dataset_name == 'wikiart':
        df = pd.read_pickle('DATA/Dataset/wikiart/wikiartINFL.pkl')
    elif dataset_name == 'fashion':
        df = pd.read_pickle('DATA/Dataset/iDesigner/idesignerINFL.pkl')
    if data_split == "all":
        artists = df['artist_name'].unique()
    else:
        artists = [data_split]
    for artist in artists:
        dataset_train = torch.load(f'DATA/Dataset_toload/{dataset_name}/Artists/{artist}_train_dataset_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}.pt')
        tripleloss_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
        net = TripletResNet_features(dataset_train.dimension).to(device)
        criterion = nn.TripletMarginLoss(margin=margin, p=2)
        optimizer = torch.optim.Adam(net.parameters(), lr =  lr)#, weight_decay = 1e-5)
        trained_model_path = f'trained_models/{dataset_name}/Artists/{artist}_TripletResNet_{feature}_{how_feature_positive}_{how_feature_negative}_{num_examples}_margin{margin}_notrans_epoch_{epochs}'
        train_artist(net,epochs, tripleloss_loader_train, criterion, optimizer, device, trained_model_path)

if __name__ == "__main__":
    start_time = time.time() 

    parser = argparse.ArgumentParser(description="Train Triplet Loss Contrastive Network on Wikiart to predict influence.")
    parser.add_argument('--dataset_name', type=str, default='wikiart', choices=['wikiart', 'fashion'])
    parser.add_argument('--feature', type=str, default='image_features', help='image_features text_features image_text_features')
    parser.add_argument('--artist_splits', action='store_true',help= 'create dataset excluding a gievn artist from training set' )
    parser.add_argument('--data_split', type=str, default = '"stratified_artists"', help= ["stratified_artists", "random_artists"])
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
    "dataset": args.dataset_name,
    "batch_size": 32,
    "epochs": 30,
    "margin": 1, 
    "num_examples": args.num_examples,
    "feature": args.feature,
    "positive_based_on_similarity": args.positive_based_on_similarity,
    "negative_based_on_similarity": args.negative_based_on_similarity,
    "data_split": args.data_split
    }

    )
    if args.artist_splits:
        main_artist(args.dataset_name, args.feature,args.data_split, args.num_examples,args.positive_based_on_similarity, args.negative_based_on_similarity)
    else:
        main(args.dataset_name,args.feature,args.data_split, args.num_examples,args.positive_based_on_similarity, args.negative_based_on_similarity)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required for training : {:.2f} seconds".format(elapsed_time))
    wandb.finish()