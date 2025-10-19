import torch
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_data(
        dataset_name: str,
        batch_size: int = 32,
        transform: transforms.Compose = transforms.ToTensor(),
        num_workers: int = 0,  # added parameter
) -> Tuple[DataLoader, DataLoader]:
    if dataset_name == 'mnist':
        train_data = datasets.MNIST(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(
            root='data', train=False, download=True, transform=transform)
    elif dataset_name == 'fashion_mnist':
        train_data = datasets.FashionMNIST(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(
            root='data', train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        train_data = datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(
            root='data', train=False, download=True, transform=transform)
    else:
        raise ValueError('Dataset not supported')

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader
