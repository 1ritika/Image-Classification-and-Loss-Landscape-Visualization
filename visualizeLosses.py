import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from common.logutils import init_log_file, append_log
from mpl_toolkits.mplot3d import Axes3D

# normalize each filter direction to unit norm
def normalize_direction(param, direction):
    if len(param.shape) > 1:
        norm = direction.reshape(direction.size(0), -1).norm(dim=1, keepdim=True)
        direction = direction.reshape(direction.size(0), -1) / (norm + 1e-10)
        direction = direction.reshape_as(param)
    else:
        direction = direction / (direction.norm() + 1e-10)
    return direction

# generate two random directions (skip batchnorm layers)
def generate_normalized_directions(model):
    d1, d2 = [], []
    for name, p in model.named_parameters():
        if p.requires_grad and "bn" not in name.lower():
            r1 = torch.randn_like(p)
            r2 = torch.randn_like(p)
            d1.append(normalize_direction(p, r1))
            d2.append(normalize_direction(p, r2))
        elif p.requires_grad:
            d1.append(torch.zeros_like(p))
            d2.append(torch.zeros_like(p))
    return d1, d2

# apply perturbation using alpha and beta directions
def perturb_model(model, original_weights, d1, d2, alpha, beta):
    with torch.no_grad():
        for p, w, a, b in zip(model.parameters(), original_weights, d1, d2):
            p.copy_(w + alpha * a + beta * b)

# compute average loss over dataset
def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
    return total_loss / total_samples

# generate 2d grid of loss values
def compute_contour(model, data_loader, criterion, device, grid_size=41, range_val=1.0):
    model.to(device)
    original_weights = [p.clone().detach() for p in model.parameters()]
    d1, d2 = generate_normalized_directions(model)

    x = np.linspace(-range_val, range_val, grid_size)
    y = np.linspace(-range_val, range_val, grid_size)
    Z = np.zeros((grid_size, grid_size))

    for i, alpha in enumerate(x):
        for j, beta in enumerate(y):
            perturb_model(model, original_weights, d1, d2, alpha, beta)
            Z[j, i] = evaluate_loss(model, data_loader, criterion, device)

    perturb_model(model, original_weights, d1, d2, 0, 0)
    return x, y, Z

# plot 2d contour map
def plot_contour(x, y, Z, title="ResNet-56", filename="plots/resnet56_contour.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    threshold = 10
    Z_clipped = np.clip(Z, Z.min(), threshold)
    Z_masked = np.ma.masked_where(Z >= threshold, Z_clipped)
    levels = np.linspace(Z_clipped.min(), threshold, 50)
    CS = ax.contour(x, y, Z_masked, levels=levels, cmap='viridis', linewidths=1)
    ax.clabel(CS, inline=True, fontsize=7, fmt="%.3f")
    ax.set_xticks(np.linspace(-1, 1, 5))
    ax.set_yticks(np.linspace(-1, 1, 5))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(f"{title} (2D)", fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# plot 3d surface
def plot_3d_surface(x, y, Z, title="(c) 3D Loss Surface", filename="plots/3d_surface.png"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', linewidth=0, antialiased=True)
    ax.set_title(f"{title}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Beta")
    ax.set_zlabel("Loss")
    ax.view_init(elev=35, azim=135)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# main function to generate plots and save xyz values
def generate_contour_plot(model, data_loader, criterion, device, title, filename):
    x, y, Z = compute_contour(model, data_loader, criterion, device)
    plot_contour(x, y, Z, title=title, filename=filename)
    plot_3d_surface(x, y, Z, title=f"{title} (3D)", filename=filename.replace("contour", "surface"))
