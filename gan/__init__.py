
from hbconfig import Config
import sys

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from . import utils
from .module import Discriminator, Generator



class GAN:

    D_PATH = f"{Config.train.model_dir}/discriminator"
    G_PATH = f"{Config.train.model_dir}/generator"

    def __init__(self):
        self.prev_step_count = 0
        self.tensorboard = utils.TensorBoard(Config.train.model_dir)

        self.discriminator = Discriminator(input=Config.model.real_dim,
                                           h1=Config.model.d_h1,
                                           h2=Config.model.d_h2,
                                           h3=Config.model.d_h3,
                                           out=1, dropout=Config.model.dropout)
        self.generator = Generator(input=Config.model.z_dim,
                                   h1=Config.model.g_h1,
                                   h2=Config.model.g_h2,
                                   h3=Config.model.g_h3,
                                   out=Config.model.real_dim)

        self.d_optimizer = None
        self.g_optimizer = None

    def train_fn(self, criterion, d_optimizer, g_optimizer, resume=True):
        self.criterion = criterion
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        if resume:
            self.prev_step_count, self.discriminator, self.d_optimizer = utils.load_saved_model(
                    self.D_PATH, self.discriminator, self.d_optimizer)
            _, self.generator, self.g_optimizer = utils.load_saved_model(
                    self.G_PATH, self.generator, self.g_optimizer)

        return self._train

    def _train(self, data_loader):
        while True:
            d_loss, g_loss = self._train_epoch(data_loader)

    def _train_epoch(self, data_loader):
        for curr_step_count, (images, _) in enumerate(data_loader):
            step_count = self.prev_step_count + curr_step_count + 1 # init value

            images = Variable(images)
            real_labels = Variable(torch.ones(images.size(0)))

            # Sample from generator
            noise = Variable(torch.randn(images.size(0), Config.model.z_dim))
            fake_images = self.generator(noise, Config.model.z_dim)
            fake_labels = Variable(torch.zeros(images.size(0)))

            # Train the discriminator
            d_loss = self._train_discriminator(
                    self.discriminator, images, real_labels, fake_images, fake_labels)

            # Sample again from the generator and get output from discriminator
            noise = Variable(torch.randn(images.size(0), Config.model.z_dim))
            fake_images = self.generator(noise, Config.model.z_dim)
            outputs = self.discriminator(fake_images, Config.model.real_dim)

            # Train the generator
            g_loss = self._train_generator(self.generator, outputs, real_labels)

            # Step Verbose & Tensorboard Summary
            if step_count % Config.train.verbose_step_count == 0:
                loss = d_loss.data[0] + g_loss.data[0]
                self._add_summary(step_count, {
                    "Loss": loss,
                    "D_Loss": d_loss.data[0],
                    "G_Loss": g_loss.data[0]
                })

                print(f"Step {step_count} - Loss: {loss} (D: {d_loss.data[0]}, G: {g_loss.data[0]})")

            # Save model parameters
            if step_count % Config.train.save_checkpoints_steps == 0:
                utils.save_checkpoint(step_count,
                                      self.D_PATH,
                                      self.discriminator,
                                      self.d_optimizer)
                utils.save_checkpoint(step_count,
                                      self.G_PATH,
                                      self.generator,
                                      self.g_optimizer)

            if step_count >= Config.train.train_steps:
                sys.exit()

        self.prev_step_count = step_count
        return d_loss, g_loss

    def _train_discriminator(self, discriminator, images, real_labels, fake_images, fake_labels):
        discriminator.zero_grad()
        real_outputs = discriminator(images, Config.model.real_dim)
        real_loss = self.criterion(real_outputs, real_labels)

        fake_outputs = discriminator(fake_images, Config.model.real_dim)
        fake_loss = self.criterion(fake_outputs, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        return d_loss

    def _train_generator(self, generator, discriminator_outputs, real_labels):
        generator.zero_grad()
        g_loss = self.criterion(discriminator_outputs, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        return g_loss

    def _add_summary(self, step, summary):
        for tag, value in summary.items():
            self.tensorboard.scalar_summary(tag, value, step)

    def evaluate_fn(self):
        pass

    def predict_fn(self):
        # Load model
        self.prev_step_count, self.generator, _ = utils.load_saved_model(
                self.G_PATH, self.generator, None)

        return self._generate_image

    def _generate_image(self, batch_size):
        noise = Variable(torch.randn(batch_size, Config.model.z_dim))
        outputs = self.generator(noise, Config.model.z_dim)

        fake_images = outputs.view(batch_size, 1, 28, 28)
        save_image(utils.denorm(fake_images.data), f"generate_images-{self.prev_step_count}.png")
        print("finished generate images..!")
