
import torch.nn as nn



class Generator(nn.Module):

    def __init__(self, input, h1, h2, h3, out):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input, h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h2, h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h3, out),
            nn.Tanh()
        )

    def forward(self, x, z_dim):
        x = x.view(x.size(0), z_dim)
        out = self.model(x)
        return out


class Discriminator(nn.Module):

    def __init__(self, input, h1, h2, h3, out, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input, h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h3, out),
            nn.Sigmoid()
        )

    def forward(self, x, real_dim):
        out = self.model(x.view(x.size(0), real_dim))
        out = out.view(out.size(0), -1)
        return out
