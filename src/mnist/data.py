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

def visualize_data(data="train", num_samples=5, images_per_row=5):
    import matplotlib.pyplot as plt
    import numpy as np

    transform = transforms.Compose([transforms.ToTensor()])
    generic_data_loader = MNISTDataLoader(transform=transform, shuffle=True)
    
    if data == "train":
        sampled_dataloader = generic_data_loader.get_train_data_loader()
    elif data == "val":
        sampled_dataloader = generic_data_loader.get_val_data_loader()

    elif data == "test":
        sampled_dataloader = generic_data_loader.get_test_data_loader()
    
    else:
        raise ValueError("The type of data you are referring to does not exist")
    

    indices = np.random.choice(len(sampled_dataloader.dataset), num_samples, replace=False)
    samples = [sampled_dataloader.dataset[i] for i in indices]

    # Plot the images
    num_rows = (num_samples + images_per_row - 1) // images_per_row

    # Plot the images in a grid layout
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 3 * num_rows))
    for i in range(num_rows):
        for j in range(images_per_row):
            index = i * images_per_row + j
            if index < num_samples:
                image, label = samples[index]
                axes[i, j].imshow(image.squeeze(), cmap='gray')
                axes[i, j].set_title(f'Label: {label}')
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')

    plt.show()

# if __name__=="__main__":
    # visualize_data("train", 20)
