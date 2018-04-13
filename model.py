
from hbconfig import Config

import torch.nn as nn
import torch.optim as optim

from gan import GAN




class Model:

    def __init__(self, mode):
        self.mode = mode

    def build(self, data_loader):
        gan = GAN()

        if self.mode == "train":
            criterion = self.build_criterion()
            d_optimizer, g_optimizer = self.build_optimizers(gan.discriminator, gan.generator)

            gan.train(data_loader, criterion, d_optimizer, g_optimizer)
        elif self.mode == "evaluate":
            gan.evaluate(data_loader)
        elif self.mode == "predict":
            gan.predict(data_loader)
        else:
            raise ValueError(f"unknown mode: {self.mode}")

    def build_criterion(self):
        return nn.BCELoss() # Binary cross entropy

    def build_optimizers(self, discriminator, generator):
        d_optimizer = optim.Adam(discriminator.parameters(),
                                 lr=Config.train.d_learning_rate,
                                 betas=Config.train.optim_betas)
        g_optimizer = optim.Adam(generator.parameters(),
                                 lr=Config.train.g_learning_rate,
                                 betas=Config.train.optim_betas)

        return d_optimizer, g_optimizer

    def build_metric(self):
        pass
