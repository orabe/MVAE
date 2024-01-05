import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler


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

        if not torch.is_tensor(image):
            image = transforms.ToTensor()(image)
        
        # one_hot_label = torch.zeros(10)
        # one_hot_label[label] = 1

        return image, label


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
            # train=True
    ):
        
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load MNIST dataset
        mnist_dataset = MNIST(root=root, train=True, download=True, transform=None)

        num_train = 50000
        num_val = 10000
        num_test = 10000
        indices = list(range(len(mnist_dataset)))

        if shuffle:
            indices = torch.randperm(len(mnist_dataset)).tolist()
        
        train_sampler = SubsetRandomSampler(indices[:num_train])
        val_sampler = SubsetRandomSampler(indices[num_train:num_train+num_val])
        test_sampler = SubsetRandomSampler(indices[num_train+num_val : num_train+num_val+num_test])

        self.train_data_loader = DataLoader(
            dataset=CustomMNIST(mnist_dataset, transform=transform, train=True),
            batch_size=batch_size,
            sampler=train_sampler
        )

        self.val_data_loader = DataLoader(
            dataset=CustomMNIST(mnist_dataset, transform=transform, train=True),
            batch_size=batch_size,
            sampler=val_sampler
        )

        self.test_data_loader = DataLoader(
            dataset=CustomMNIST(mnist_dataset, transform=transform, train=False),
            batch_size=batch_size,
            sampler=test_sampler
        )

    def get_train_data_loader(self):
        return self.train_data_loader

    def get_val_data_loader(self):
        return self.val_data_loader

    def get_test_data_loader(self):
        return self.test_data_loader

# Usage:
# transform = transforms.Compose([transforms.ToTensor()])
# mnist_data_loader = MNISTDataLoader(transform=transform, shuffle=True)

# train_loader = mnist_data_loader.get_train_data_loader()
# val_loader = mnist_data_loader.get_val_data_loader()
# test_loader = mnist_data_loader.get_test_data_loader()
