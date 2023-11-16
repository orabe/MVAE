import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.dataset import MNIST

class CustomMNIST(Dataset):
    def __init__(
            self, 
            mnist_dataset, 
            transform=None
    ):
        self.mnist_dataset = mnist_dataset
        self.transform = transform

    def __len__(self):
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class MNISTDataLoader:
    def __init__(
            self,
            root="./data",
            batch_size=64,
            transform=None,
            shuffle=True
    ):
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load MNIST dataset
        mnist_dataset = MNIST(root=root, train=True, download=True, transform=transform)

        # Create custom dataset
        self.custom_dataset = CustomMNIST(mnist_dataset, transform=transform)

        self.data_loader = DataLoader(
            dataset=self.custom_dataset,
            batch_size=batch_size
            shuffle=shuffle
        )

    def get_data_loader(self):
        return self.data_loader


