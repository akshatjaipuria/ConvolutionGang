import torchvision
import torch
from albumentations import *
from albumentations.pytorch import ToTensorV2
import numpy as np


class train_transforms:
    def __init__(self):
        self.train_transform = Compose([
            Resize(64, 64, 3),
            Rotate(),
            PadIfNeeded(min_height=70, min_width=70,),
            RandomCrop(64, 64,),
            HorizontalFlip(),
            CoarseDropout(max_holes=1, max_height=16, max_width=16),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.train_transform(image=img)['image']
        return img


class test_transforms:
    def __init__(self):
        self.test_transform = Compose([
            Resize(64, 64, 3),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.test_transform(image=img)['image']
        return img


def tiny_imagenet():
    trainset = torchvision.datasets.ImageFolder(root="./tiny-imagenet-200/train_set/", transform=train_transforms())
    testset = torchvision.datasets.ImageFolder(root="./tiny-imagenet-200/test_set/", transform=test_transforms())

    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4,
                                              pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, testloader
