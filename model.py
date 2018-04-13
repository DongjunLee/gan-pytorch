
from hbconfig import Config

import torch.nn as nn
import torch.optim as optim

from gan import GAN




class Model:

    def __init__(self, mode):
        self.mode = mode

    def build_fn(self):
        gan = GAN()

        if self.mode == "train":
            criterion = self.build_criterion()
            d_optimizer, g_optimizer = self.build_optimizers(gan.discriminator, gan.generator)

            return gan.train_fn(criterion, d_optimizer, g_optimizer)
        elif self.mode == "evaluate":
            return gan.evaluate_fn
        elif self.mode == "predict":
            return gan.predict_fn
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
