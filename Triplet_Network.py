import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
from torch.utils.data import DataLoader, Dataset
from create_data_loader import TripletLossDataset_features
import time 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'




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
    


def train(model,epochs, train_loader, val_loader, criterion, optimizer, device,name_model):
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
 

    if os.path.exists('trained_models') == False:
        os.makedirs('trained_models')
    torch.save(model.state_dict(), f'trained_models/{name_model}.pth')

    plt.plot(loss_plot_train, label='Training Loss')
    plt.plot(loss_plot_val, label='Validation Loss')
    plt.plot(accuracy_plot_train, label='Training Accuracy')
    plt.plot(accuracy_plot_val, label='Validation Accuracy')
    plt.title('Model Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    if os.path.exists('trained_models/Loss_figures') == False:
        os.makedirs('trained_models/Loss_figures')
    plt.savefig(f'trained_models/Loss_figures/tripletloss_{name_model}.png')


    

def main():    
    epochs = wandb.config.epochs
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    feature = 'image_features'
    dataset_train = torch.load(f'DATA/Dataset_toload/train_dataset_{feature}_all.pt')
    dataset_val = torch.load(f'DATA/Dataset_toload/val_dataset_{feature}_all.pt')
    tripleloss_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    tripleloss_loader_val = DataLoader(dataset_val, shuffle=False, batch_size=batch_size)
    net = TripletResNet_features(dataset_train.dimension).to(device)

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(net.parameters(), lr =  lr)
    train(net,epochs, tripleloss_loader_train, tripleloss_loader_val, criterion, optimizer, device, f'TripletResNet_{feature}')

if __name__ == "__main__":
    start_time = time.time() 
    wandb.init(
    # set the wandb project where this run will be logged
    project="Triplet_Network_Wikiart_predict_Influence",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0005,
    "architecture": "Triplet Network",
    "dataset": "Wikiart 1000",
    "preprocessing": "ResNet34",
    "batch_size": 32,
    "epochs": 10,
    }
    )
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print("Time required for training : {:.2f} seconds".format(elapsed_time))
    wandb.finish()