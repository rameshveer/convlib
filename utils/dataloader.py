from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np


def dataloader(size, workers, cuda, train_transform_param):
    
    train_transform = transforms.Compose([transforms.ToTensor(), train_transform_param])
    
    test_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=train_transform)

    testset = datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=test_transform)

    if cuda:
        size = size
    else:
        size = size

    dataloader_args = dict(shuffle=True, batch_size=size, num_workers=workers)

    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return trainloader, testloader


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

