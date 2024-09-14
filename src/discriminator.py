import torch
from torch import nn
import os

class Discriminator(nn.Module):
    def __init__(self, flat_img_size, leaky_relu = 0.2):
        
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(flat_img_size, 512),
            nn.LeakyReLU(leaky_relu, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(leaky_relu, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(leaky_relu, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
    def save(self, path):
        ''' path relative to script directory '''
        curr_dir = os.getcwd()
        full_path = os.path.join(curr_dir, path)
        torch.save(self.state_dict(), full_path)
    
    def load(self, path):
        curr_dir = os.getcwd()
        full_path = os.path.join(curr_dir, path)
        self.load_state_dict(torch.load(full_path, weights_only=True))