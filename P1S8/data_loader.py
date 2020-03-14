import torchvision
import torch
import torchvision.transforms as transforms


def cifar10():
        train_transforms = transforms.Compose([
            #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
            #  transforms.RandomApply([transforms.CenterCrop(22),], p=0.1),
            #  transforms.Pad(padding=3, fill=(1,), padding_mode='constant'),
             transforms.RandomRotation((-7.0, 7.0), fill=(1, 1, 1)),
            transforms.RandomHorizontalFlip(),
            #  transforms.Resize((32, 32, 3)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159)),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
            #  Note the difference between (0.1307) and (0.1307,)
        ])

        # Test Phase transformations
        test_transforms = transforms.Compose([
            #  transforms.Resize((28, 28)),
            #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159))
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

        SEED = 1

        # CUDA?
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)

        # For reproducibility
        torch.manual_seed(SEED)

        if cuda:
            torch.cuda.manual_seed(SEED)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4,
                                                  pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return trainloader, testloader, classes
