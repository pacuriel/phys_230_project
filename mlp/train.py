"""
File to train MLP model.
Pablo Curiel
May 2025
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os

from mlp import MLP
from met_dataset import METDataset

# Dataset
DATA_DIR = "/home/pcuriel/data/phys_project/phys_230_project/sim_data"
ALL_FILES = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith(".npz")]

# Hyperparameters
TRAIN_SPLIT = 0.7
TEST_SPLIT = 1 - TRAIN_SPLIT
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 33
exp_id = datetime.now().strftime("%d-%b-%Y_%H%M_%S") + f"_{seed}_{NUM_EPOCHS}"
exp_dir = os.path.join("experiments", exp_id)

def plot_losses(train_losses: list, test_losses: list, num_epochs: int) -> None:
    """Function to plot the train and test losses per epoch."""
    plt.plot(list(range(num_epochs)), train_losses, label="train loss")
    plt.plot(list(range(num_epochs)), test_losses, label="test loss")
    plt.legend()
    plt.grid() #GRID ON OR OFF????
    plt.title("Train/Test Loss per Epoch")
    plt.xlabel("Epochs")
    plt.xlim((0, num_epochs))
    plt.ylabel(r"Loss $L$")
    plt.savefig(os.path.join(exp_dir, f'train_test_loss.png'))    
    plt.clf()

def test(model, test_loader, loss_fcn):
    model.eval()
    test_loss = 0.0
    num_batches = len(test_loader)

    with torch.no_grad():
        # Loop over all batches in data loader
        for batch_idx, (input_batch, gt_batch, indices) in enumerate(test_loader):
            # Putting tensors on GPU
            input_batch = input_batch.to(device)
            gt_batch = gt_batch.to(device)

            preds = model(input_batch) # Getting model predictions
            loss = loss_fcn(preds, gt_batch) # Calculating loss
            test_loss += loss.item() 

        test_loss /= num_batches

    model.train()
    return test_loss

def train(model, train_loader, test_loader, loss_fcn, optimizer):
    
    num_batches = len(train_loader)
    train_losses = []
    test_losses = []
    
    best_train_loss = float('inf')
    best_test_loss = float('inf')

    # Loop over each epoch
    for epoch in range(NUM_EPOCHS):
        model.train() # Train mode
        epoch_loss = 0.0 # Reset epoch loss
        data_loader_loop = tqdm(train_loader)

        # Loop over all batches in data loader
        for batch_idx, (input_batch, gt_batch, indices) in enumerate(data_loader_loop):
            # Putting tensors on GPU
            input_batch = input_batch.to(device)
            gt_batch = gt_batch.to(device)

            preds = model(input_batch) # Getting model predictions
            loss = loss_fcn(preds, gt_batch) # Calculating loss

            epoch_loss += loss.item() 

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_loader_loop.set_postfix(loss=(epoch_loss / num_batches)) #updating tqdm bar to show loss 

        epoch_loss /= num_batches #averaging loss by number of batches
        train_losses.append(epoch_loss) #storing loss

        if epoch_loss < best_train_loss: 
            best_train_loss = epoch_loss #updating best loss
            torch.save(model, os.path.join(exp_dir, 'best_train_model.pt')) #saving model

        test_loss = test(model, test_loader, loss_fcn)
        test_losses.append(test_loss)

        # Updating best test loss
        if test_loss < best_test_loss: 
            best_test_loss = test_loss
            torch.save(model, os.path.join(exp_dir, 'best_test_model.pt')) #saving model

        print(f"Epoch: {epoch + 1} / {NUM_EPOCHS} \t Train loss: {epoch_loss} \t Best train Loss: {best_train_loss} \t Test loss: {test_loss} \t Best test Loss: {best_test_loss}\n")

    # Plotting loss curves
    plot_losses(train_losses=train_losses, test_losses=test_losses, num_epochs=NUM_EPOCHS)

def main():
    # Splitting files for training/testing
    
    train_files, test_files = train_test_split(ALL_FILES, test_size=TEST_SPLIT, random_state=seed) 
    
    # Storing train/test datasets and dataloaders
    train_set = METDataset(file_paths=train_files)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_set = METDataset(file_paths=test_files)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Loading in model, loss function, and optimizer
    model = MLP(); model = model.to(device)
    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    os.makedirs(exp_dir, exist_ok=True)

    train(model=model,
          train_loader=train_loader,
          test_loader=test_loader,
          loss_fcn=loss_fcn, 
          optimizer=optimizer)
    

if __name__ == "__main__":
    main()