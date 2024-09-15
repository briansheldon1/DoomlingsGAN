import os
import torch
from torchvision import transforms
from PIL import Image

def load_image(path):
    ''' 
        Take in a path to png image - return flattened norm tensor of rgb image 

        Arguments:
            - path: path should be relative to src directory 
              ("../traits_images/ex.png")
    '''
    # get full path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(curr_dir, path)

    # open image
    img = Image.open(full_path).convert("RGB")

    # define transformations
    transform = transforms.Compose([
        transforms.Resize((80, 112)),
        transforms.ToTensor(),
        torch.flatten
    ])

    img_tensor = transform(img)

    return img_tensor


def load_dir_images(rel_dir_path):

    # get full path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    full_dir_path = os.path.join(curr_dir, rel_dir_path)


    img_tensors = []

    for filename in os.listdir(full_dir_path):
        if filename.endswith('.png'):
            img_tensor = load_image(os.path.join(rel_dir_path, filename))
            img_tensors.append(img_tensor)


    return torch.stack(img_tensors, dim=0)

def wass_disc_loss(real_out, fake_out):
    ''' 
        Wasserstein loss for discriminator 
        Arguments:
            real_out: labels of discriminator(data) where 1 represents a 
                      prediction of data and 0 prediction of fake data

            fake_out: labels of discriminator(generator(noise)) where 1 
                      represents pred of data, 0 pred of fake data

        Returns:
            loss: Wasserstein loss for discr over this batch of data
    '''
    return -(torch.mean(real_out) - torch.mean(fake_out) )

def wass_gen_loss(fake_out):
    ''' 
        Wasserstein loss for generator

        Arguments:
            fake_out: labels of discriminator(generator(noise)) where 1 
                      represents pred of data, 0 pred of fake data

        Returns:
            loss: Wasserstein loss for discr over this batch of data
    '''
    return -torch.mean(fake_out)



def clip_weights(model: torch.nn.Module, clip_value=0.01):
    for param in model.parameters():
        param.data.clamp_(-clip_value, clip_value)