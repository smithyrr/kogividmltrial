#This script performs the following tasks:

#Imports the necessary libraries and modules for deep learning, data handling, and logging with WandB.
#Initializes a new WandB run under the project "self-learning-video-generator."
#Defines the data path and transformation for the UCF101 dataset (resizing and converting images to tensors).
#Creates the UCF101 dataset and splits it into training and validation sets.
#Creates DataLoader instances for the training and validation sets.
#Defines the input size, hidden size, and latent size for the VAE (Variational Autoencoder) model.
#Creates the VAE model and moves it to the available device (CUDA or CPU).
#Defines the optimizer as Adam with a learning rate of 1e-3.
#Sets up WandB to watch the model, logging all parameters and gradients every 100 steps.
#Initializes the training loop, including early stopping logic based on validation loss.
#In each epoch:
#a. Trains the model on the training set.
#b. Logs the gradients histograms to WandB.
#c. Validates the model on the validation set.
#d. Logs the reconstructed images to WandB every 10 epochs.
#e. Updates the early stopping counter and saves the best model based on validation loss.
#If early stopping is triggered, the training loop breaks and the script ends.
#Finishes the WandB run, logging all final results and closing the connection.
#This script trains a VAE model on the UCF101 dataset, monitoring training and validation losses, gradients, and reconstructed images using WandB. The early stopping mechanism helps to prevent overfitting by stopping training when the validation loss stops improving.



import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import UCF101
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Lambda
import wandb
import numpy as np
from torchvision.utils import make_grid
from kogvidmodel import VAE

wandb.init(project='self-learning-video-generator')

data_path = '/home/cognitron/kogivid/UCF1011/UCF-101'
annotation_path = '/home/cognitron/kogivid/ucfTrainTestlist'

def collect_video_files(data_path):
    video_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mkv', '.flv', '.mov')):
                video_list.append(os.path.join(root, file))
    return video_list

# Define your custom transformation function for video frames

# Define your custom transformation function for video frames
def video_transform(video_frames):
    transformed_frames = []
    for frame in video_frames:
        transformed_frame = transform(frame)
        transformed_frames.append(transformed_frame)
    return torch.stack(transformed_frames)

# Set the desired number of frames per clip (e.g., 16)
frames_per_clip = 16
step_between_clips = 1
num_workers = 4

dataset = UCF101(data_path, frames_per_clip, num_workers=num_workers, step_between_clips=step_between_clips, annotation_path=annotation_path, transform=Compose([Lambda(video_transform)]))


video_list = collect_video_files(data_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define your custom transformation function for video frames

train_len = int(len(dataset) * 0.8)
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=2)

input_size = 64 * 64 * 3
hidden_size = 512
latent_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(input_size, hidden_size, latent_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

wandb.watch(model, log="all", log_freq=100)

wandb.watch(model, log_freq=100)

from torchvision.datasets.video_utils import VideoClips
from torchvision.transforms import Compose, Lambda



def loss_function(x_recon, x, mu, log_var):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + kld_loss) / x.size(0)


num_epochs = 100
patience = 5
best_val_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_grads = []

    for x, _ in train_loader:
        x = x.to(device)

        optimizer.zero_grad()

        x_recon, mu, log_var = model(x)
        loss = loss_function(x_recon, x, mu, log_var)
        epoch_loss += loss.item()

        loss.backward()

        # Store gradients for histogram logging
        for name, param in model.named_parameters():
            if param.grad is not None:
                epoch_grads.append(param.grad.view(-1).cpu().numpy())

        optimizer.step()

        average_loss = epoch_loss / len(train_loader)
    wandb.log({'epoch': epoch, 'train_loss': average_loss}, step=epoch)

    # Log gradients histograms
    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"{name}_grad_hist": wandb.Histogram(param.grad.view(-1).cpu().numpy())}, step=epoch)

    average_loss = epoch_loss / len(train_loader)
    
    wandb.log({'epoch': epoch, 'train_loss': average_loss}, step=epoch)

    # Validate the model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            x_recon, mu, log_var = model(x)
            loss = loss_function(x_recon, x, mu, log_var)
            val_loss += loss.item()

        # Log validation images
        if epoch % 10 == 0:  # Change the frequency of logging images as needed
            x_grid = make_grid(x_recon[:8].cpu(), nrow=4)  # Change the number of images and layout as needed
            wandb.log({"reconstructed_images": wandb.Image(x_grid)}, step=epoch)

    val_loss /= len(val_loader)
    wandb.log({'epoch': epoch, 'val_loss': val_loss}, step=epoch)

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"best_model.pth")
        wandb.save(f"best_model.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    model.train()

wandb.finish()


