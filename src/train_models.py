import torch
import torch.optim as optim

from generator import Generator
from discriminator import Discriminator


def train_Main(epochs=50):

    # initialize models
    generator = Generator()
    discriminator = Discriminator()

    # create optimizers
    opt_G = optim.Adam(generator.parameters(), lr=0.0001)
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0001)

    for epoch in range(epochs):
        pass



if __name__ == '__main__':
    pass


