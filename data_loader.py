
from hbconfig import Config

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms



def make_data_loader(mode, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    is_train = mode == "train"

    mnist_dataset = dsets.MNIST(root=Config.data.path,
                                train=is_train,
                                download=True,
                                transform=transform)
    return torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=shuffle)
