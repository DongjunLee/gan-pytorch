#!/usr/bin/env python

from hbconfig import Config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data_loader import make_data_loader
from model import Discriminator, Generator


generator = Generator(input=Config.data.z_dim,
                      h1=Config.model.g_h1,
                      h2=Config.model.g_h2,
                      h3=Config.model.g_h3,
                      out=Config.data.real_dim)
discriminator = Discriminator(input=Config.data.real_dim,
                              h1=Config.model.d_h1,
                              h2=Config.model.d_h2,
                              h3=Config.model.d_h3,
                              out=1, dropout=Config.model.dropout)

criterion = nn.BCELoss()  # Binary cross entropy
d_optimizer = optim.Adam(discriminator.parameters(), lr=Config.train.d_learning_rate, betas=Config.train.optim_betas)
g_optimizer = optim.Adam(generator.parameters(), lr=Config.train.g_learning_rate, betas=Config.train.optim_betas)

def train_discriminator(discriminator, images, real_labels, fake_images, fake_labels):
    discriminator.zero_grad()
    outputs = discriminator(images, Config.data.real_dim)
    real_loss = criterion(outputs, real_labels)
    real_score = outputs

    outputs = discriminator(fake_images, Config.data.real_dim)
    fake_loss = criterion(outputs, fake_labels)
    fake_score = outputs

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss, real_score, fake_score

def train_generator(generator, discriminator_outputs, real_labels):
    generator.zero_grad()
    g_loss = criterion(discriminator_outputs, real_labels)
    g_loss.backward()
    g_optimizer.step()
    return g_loss


data_loader = make_data_loader("train", Config.train.batch_size)

for epoch in range(Config.train.num_epochs):
    for n, (images, _) in enumerate(data_loader):
        images = Variable(images)
        real_labels = Variable(torch.ones(images.size(0)))

        # Sample from generator
        noise = Variable(torch.randn(images.size(0), Config.data.z_dim))
        fake_images = generator(noise, Config.data.z_dim)
        fake_labels = Variable(torch.zeros(images.size(0)))

        # Train the discriminator
        d_loss, real_score, fake_score = train_discriminator(discriminator, images, real_labels, fake_images, fake_labels)

        # Sample again from the generator and get output from discriminator
        noise = Variable(torch.randn(images.size(0), Config.data.z_dim))
        fake_images = generator(noise, Config.data.z_dim)
        outputs = discriminator(fake_images, Config.data.real_dim)

        # Train the generator
        g_loss = train_generator(generator, outputs, real_labels)

    if epoch % Config.train.print_interval == 0:
        print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, '
                  'D(x): %.2f, D(G(z)): %.2f'
                  %(epoch + 1, Config.train.num_epochs, n+1, Config.train.batch_size, d_loss.data[0], g_loss.data[0],
                    real_score.data.mean(), fake_score.data.mean()))
