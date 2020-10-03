from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class dataloader(DataLoader):

    def dataloader(size, workers, cuda):

        trainset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transforms.transform)
        testset = datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transforms.transform)

        if cuda:
            size = 128
        else:
            size = 64

        dataloader_args = dict(shuffle=True, batch_size=size, num_workers=workers)

        trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

        testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

        return trainloader, testloader


    def imgshow(self):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
