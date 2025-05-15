import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Visualization and I/O
import imageio
import matplotlib.pyplot as plt

# Internal Modules
from dataset import get_rays
from rendering import rendering
from model import Voxels, Nerf
from ml_helpers import training


def prepare_dataloader(o, d, rgb, batch_size=1024, crop_region=None):
    """
    Creates a dataloader from origin, direction, and RGB target tensors.
    Optionally crops the central region if crop_region is specified.
    """
    if crop_region:
        h_start, h_end, w_start, w_end = crop_region
        o = o.reshape(90, 400, 400, 3)[:, h_start:h_end, w_start:w_end, :].reshape(-1, 3)
        d = d.reshape(90, 400, 400, 3)[:, h_start:h_end, w_start:w_end, :].reshape(-1, 3)
        rgb = rgb.reshape(90, 400, 400, 3)[:, h_start:h_end, w_start:w_end, :].reshape(-1, 3)
    else:
        o = o.reshape(-1, 3)
        d = d.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3)

    data = torch.cat([
        torch.from_numpy(o).float(),
        torch.from_numpy(d).float(),
        torch.from_numpy(rgb).float()
    ], dim=1)

    return DataLoader(data, batch_size=batch_size, shuffle=True)


def main():
    # === Configuration ===
    data_path = '/home/thaoanh/Documents/TA/selfLearn/NeRF/fox/fox'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 1024
    tn, tf = 8.0, 12.0
    nb_epochs = 10
    lr = 1e-3
    gamma = 0.5
    nb_bins = 100
    lr_milestones = [5, 10]

    # === Load Data ===
    o, d, rgb = get_rays(data_path, mode='train')
    test_o, test_d, test_rgb = get_rays(data_path, mode='test')

    # Dataloaders
    dataloader = prepare_dataloader(o, d, rgb, batch_size=batch_size)
    crop_region = (100, 300, 100, 300)
    dataloader_warmup = prepare_dataloader(o, d, rgb, batch_size=batch_size, crop_region=crop_region)

    # === Model Setup ===
    model = Nerf(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)

    # === Training ===
    print("Starting full training...")
    training_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, dataloader, device=device)

    # === Save Model ===
    torch.save(model.cpu(), 'model_nerf')
    print("Model saved as 'model_nerf'")


if __name__ == "__main__":
    main()
