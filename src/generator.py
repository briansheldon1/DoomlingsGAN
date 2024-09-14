import torch
import torch.nn as nn
import os

class Generator(nn.Module):
    def __init__(self, flat_img_size, rand_vec_size = 100, leaky_relu = 0.2):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(rand_vec_size, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, flat_img_size),
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