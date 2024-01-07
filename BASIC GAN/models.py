import torch
import torch.nn as nn
import torchvision


# DISCRIMINATOR MODEL
"""
img_dim: image dimension(exp->28x28=784) of the input image which you want to classify as true or false 
"""


class Discriminator(nn.Module):
    def __init__(self, img_dim: int):
        super(Discriminator, self).__init__()

        # model architecture
        self.model = nn.Sequential(
            nn.Linear(in_features=img_dim, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# GENERATOR MODEL
"""
z_dim = dimension of your latent noise (random noise)
img_dim = generate image shape
"""


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim: int):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.1), nn.Linear(256, img_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
