import torchvision
import torch
from albumentations import *
from albumentations.pytorch import ToTensorV2
import numpy as np


class train_transforms:
    def __init__(self):
        self.train_transform = Compose([
            Resize(32, 32, 3),
            HorizontalFlip(p=.4),
            CoarseDropout(max_holes=1, max_height=16, max_width=16),
            Normalize(mean=[0.4914, 0.48216, 0.44653], std=[0.24703, 0.24349, 0.26159]),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.train_transform(image=img)['image']
        return img


class test_transforms:
    def __init__(self):
        self.test_transform = Compose([
            Resize(32, 32, 3),
            Normalize(mean=[0.4914, 0.48216, 0.44653], std=[0.24703, 0.24349, 0.26159]),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.test_transform(image=img)['image']
        return img


def cifar10():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms())
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms())

    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0,
                                              pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes
