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

    # open image
    img = Image.open(path).convert("RGB")

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
            img_tensor = load_image(os.path.join(full_dir_path, filename))
            img_tensors.append(img_tensor)


    return torch.stack(img_tensors, dim=0)
