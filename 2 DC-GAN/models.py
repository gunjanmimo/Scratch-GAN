import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_channel: int, img_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channel,
                out_channels=img_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            self.block(
                in_channels=img_dim,
                out_channels=img_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self.block(
                in_channels=img_dim * 2,
                out_channels=img_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self.block(
                in_channels=img_dim * 4,
                out_channels=img_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=img_dim * 8,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, z_dim: int, num_channel: int, feature_g: int) -> None:
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            self.block(
                in_channels=z_dim,
                out_channels=feature_g * 16,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            self.block(
                in_channels=feature_g * 16,
                out_channels=feature_g * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self.block(
                in_channels=feature_g * 8,
                out_channels=feature_g * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self.block(
                in_channels=feature_g * 4,
                out_channels=feature_g * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ConvTranspose2d(
                in_channels=feature_g * 2,
                out_channels=num_channel,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


def initialize_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == "__main__":
    N, num_channel, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, num_channel, H, W))
    disc = Discriminator(num_channel=num_channel, img_dim=8)
    initialize_weight(disc)
    assert disc(x).shape == (N, 1, 1, 1)

    gen = Generator(z_dim=z_dim, num_channel=num_channel, feature_g=198)
    initialize_weight(gen)
    z = torch.randn((N, z_dim, 1, 1))
    print(gen(z).shape)
    assert gen(z).shape == (N, num_channel, H, W)
    print("success")
