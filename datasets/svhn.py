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

    # test_dataset = datasets.SVHN(root='/home/hhjung/hhjung/SVHN/', split='test', 
    #                                     transform=transforms.Compose([transforms.ToTensor()
    #                                         , transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
    #                                     download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=False)

    if train:
        return train_loader
    else:
        return test_loader