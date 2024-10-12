import torch
from PIL import Image, ImageEnhance
import numpy as np

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
    return torch.mean(fake_out) - torch.mean(real_out)


def wass_disc_gp_loss(descriminator, real_data, fake_data, device='cpu'):
    ''' 
        Get gradient penalty for discriminator 
        Arguments:
            descriminator: descriminator model
            real_data: tensor of real images data
            fake_data: tensor of fake images data
            device: device to run on ('cpu' or 'cuda')
    '''

    # get batch size
    batch_size = real_data.size(0)

    # interpolate between real and fake data
    eps_shape = [batch_size] + [1]*(len(real_data.shape)-1)
    eps = torch.randn(eps_shape, device=device)
    interp = eps*real_data + (1-eps)*fake_data
    interp_out = descriminator(interp)

    # get gradient of output
    grad = torch.autograd.grad(
            outputs=interp_out, 
            inputs=interp, 
            grad_outputs=torch.ones_like(interp_out, device=device),
            create_graph = True,
            retain_graph = True,
            only_inputs = True, 
            allow_unused = True)[0]
    

    # calculate penalty as how far norm is from 1
    penalty = ((grad.norm(2, dim=1)-1)**2).mean()

    return penalty


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


def normalize_noise(noise):
    ''' Normalize noise to have a mean of 0 and standard deviation of 1 '''

    # Normalize the noise to have a mean of 0 and standard deviation of 1
    mean = noise.mean(dim=1, keepdim=True)
    std = noise.std(dim=1, keepdim=True) + 1e-8  # Adding small epsilon to avoid division by zero
    noise_normalized = (noise - mean) / std
    return noise_normalized


def adjust_contrast_based_on_brightness(image: Image.Image) -> Image.Image:
    ''' Adjust contrast based on brightness of the image '''
    # Convert image to grayscale to calculate average brightness
    grayscale_image = image.convert("L")  # "L" mode is for grayscale
    image_np = np.array(grayscale_image)

    # Calculate the average brightness
    avg_brightness = np.mean(image_np)

    # Dynamically determine contrast factor based on brightness
    # If brightness is high, reduce contrast, and vice versa
    contrast_factor = -0.04 * avg_brightness + 7
    contrast_factor = max(0.75, min(contrast_factor, 2.5))

    # Apply contrast adjustment
    enhancer = ImageEnhance.Contrast(image)
    adjusted_image = enhancer.enhance(contrast_factor)
    
    return adjusted_image