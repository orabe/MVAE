import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


"""
    This class is responsible for the interface of the function and the dataset.
"""
class CustomMNIST(Dataset):
    def __init__(
            self, 
            mnist_dataset, 
            transform=None,
            train=True
    ):
        self.mnist_dataset = mnist_dataset
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]

        # print(self.transform)
        if not torch.is_tensor(image):
            image = transforms.ToTensor()(image)
        
        one_hot_label = torch.zeros(10)
        one_hot_label[label] = 1

        return image, one_hot_label


"""
    This class is the dataloader for the MVAE model. 
    It takes the custom MNIST dataset for training and testing and creates the dataloader
    USAGE:
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train_data_loader = MNISTDataLoader(transform=transform, train=True)
    mnist_test_data_loader = MNISTDataLoader(transform=transform, train=False)

"""
class MNISTDataLoader:
    def __init__(
            self,
            root="./data",
            batch_size=64,
            transform=None,
            shuffle=True,
            train=True
    ):
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train

        # Load MNIST dataset
        mnist_dataset = MNIST(root=root, train=train, download=True, transform=None)

        # Create custom dataset
        self.custom_dataset = CustomMNIST(mnist_dataset, transform=transform, train=train)

        self.data_loader = DataLoader(
            dataset=self.custom_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def get_data_loader(self):
        return self.data_loader

