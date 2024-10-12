import torch
from torch import nn
import os

''' Discriminator model which is a CNN with 2 layers and 2 dense layers '''
class Discriminator(nn.Module):

    def __init__(self, img_shape, leaky_relu = 0.2):
        
        # parent initilizaiton
        super().__init__()
        
        # unpack image shape
        channels, width, height = img_shape

        # define model
        self.net = nn.Sequential(

            # first cnn (divides image size in 2)
            nn.Conv2d(channels, 32, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(leaky_relu, inplace=True),

            # second cnn (divides images size in 2)
            nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(leaky_relu, inplace=True),

            # Flatten output
            nn.Flatten(),

            # First dense layer to 128 nodes
            nn.Linear(4*width*height, 128),
            nn.LeakyReLU(leaky_relu, inplace=True),

            # Second dense layer to 1 node
            nn.Linear(128, 1),
            nn.LeakyReLU(leaky_relu, inplace=True)
        )

    def forward(self, x):
        ''' forward pass network '''
        return self.net(x)
    
    def save(self, path):
        '''
            Save model as pkl file
            
            Arguments:
                - path: path to save model relative to this file

        '''

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(curr_dir, path)
        torch.save(self.state_dict(), full_path)
    
    def load(self, path):
        '''
            Load model as pkl file
            
            Arguments:
                - path: path to saved model relative to this file

        '''
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(curr_dir, path)
        self.load_state_dict(torch.load(full_path, weights_only=True))


''' Generator model which is a CNN with 2 layers and 1 dense layer '''
class Generator(nn.Module):

    def __init__(self, img_shape, noise_size = 100):
        ''' 
            Arguments:
                - img_shape: shape of the image (channels, width, height)
                - noise_size: size of the noise vector
        '''

        # parent initilizaiton
        super().__init__()

        # unpack image shape
        channels, width, height = img_shape

        # define model
        self.net = nn.Sequential(

            # Layer 1:  Noise(100 x 1 x 1) -> (1024 x 4 x 4)
            nn.ConvTranspose2d(noise_size, 1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            # Layer 2: (1024 x 4 x 4) -> (512 x 8 x 8)
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Flatten and final linear layer to correct flattened size
            nn.Flatten(),
            nn.Linear(512*8*8, channels*width*height),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        ''' forward pass network '''
        return self.net(x)
    
    def save(self, path):
        '''
            Save model as pkl file
            
            Arguments:
                - path: path to save model relative to this file

        '''
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(curr_dir, path)
        torch.save(self.state_dict(), full_path)
    
    def load(self, path):
        '''
            Load model as pkl file
            
            Arguments:
                - path: path to saved model relative to this file

        '''
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(curr_dir, path)
        self.load_state_dict(torch.load(full_path, weights_only=True))