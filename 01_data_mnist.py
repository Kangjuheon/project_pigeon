import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist_dataloaders(batch_size=64, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    train_size = int(len(mnist_full) * 5 / 6)
    val_size = len(mnist_full) - train_size
    train_dataset, val_dataset = random_split(mnist_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader 