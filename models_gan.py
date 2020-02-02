import torch
from torch import nn

class Generator(nn.Module):
  """An image generator that takes input of any size and outputs a 64x64 image with the given number of channels."""

  def __init__(self, in_size, out_channels):
    super().__init__()

    self.first_lin = nn.Linear(in_features=in_size, out_features=4*4*128)
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(num_features=64),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(num_features=64),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(num_features=32),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(32, out_channels, 3, stride=2, padding=1, output_padding=1),
        nn.Tanh(),
    )

  def forward(self, x):
    a = self.first_lin(x)
    a = a.reshape(-1, 128, 4, 4)
    y = self.decoder(a)
    # since we are using Tanh as final activation like suggested by DCGAN paper,
    # we should shift outputs from [-1, 1] back to [0, 1].
    y = 0.5 * y + 0.5
    # change from (N, C, W, H) to (N, W, H, C)
    y = y.permute(0, 2, 3, 1)
    return y


# Adapted from: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
class Discriminator(nn.Module):
  """A conditional discriminator that outputs if an input image is both real and fits the conditioning input."""
  
  def __init__(self, image_size, in_channels, condition_dim):
    super().__init__()

    def discriminator_block(in_filters, out_filters, bn=True):
        block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        return block

    # The height and width of a downsampled image.
    ds_size = image_size // 2 ** 4

    self.image_enc = nn.Sequential(
        *discriminator_block(in_channels, 32, bn=False),
        *discriminator_block(32, 64),
        *discriminator_block(64, 64),
        *discriminator_block(64, 128),
        nn.Flatten(),
        nn.Linear(128 * ds_size ** 2, 64),
        nn.LeakyReLU(0.2, inplace=True),
    )

    self.final_processor = nn.Sequential(
        nn.Linear(64 + condition_dim, 64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )

  def forward(self, img, condition_in):
    # change from (N, W, H, C) to (N, C, W, H)
    img = img.permute(0, 3, 1, 2)
    out = self.image_enc(img)
    out = torch.cat([out, condition_in], 1)
    out = self.final_processor(out)
    return out
