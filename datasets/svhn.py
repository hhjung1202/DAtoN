"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms

import params

def get_svhn(train):
    print("SVHN Data Loading ...")

    train_dataset = datasets.SVHN(root='/home/hhjung/hhjung/SVHN/', split='train', 
                                        transform=transforms.Compose([transforms.ToTensor()
                                            , transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                                        download=True)

    test_dataset = datasets.SVHN(root='/home/hhjung/hhjung/SVHN/', split='test', 
                                        transform=transforms.Compose([transforms.ToTensor()
                                            , transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                                        download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=False)

    if train:
        return train_loader
    else:
        return test_loader

def get_mnist(train):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return mnist_data_loader
