import torch
import torch.optim as optim
import numpy as np

from models import Generator, Discriminator
from augmented_dataset import AugmentedDataset
from util import wass_disc_loss, wass_disc_gp_loss, wass_gen_loss

'''
    Main file for training the discriminator and generator
    This uses Wassertein loss and RMSprop optimizer
'''

def train_Main(epochs=50, batch_size=100, 
               save_gen_path = None, save_discr_path = None, verbose=False):
    ''' 
        Main function for training models
        Arguments:
            - epochs: number of epochs to train
            - batch_size: size of each batch
            - save_gen_path: path to save generator model
            - save_discr_path: path to save discriminator model
            - verbose: boolean of whether print out loss at each epoch
    '''

    # check if cuda is available
    if (torch.cuda.is_available()):
        device = 'cuda'
    else:
        device = 'cpu'

    # load real images, use augmented dataset to apply augmentations
    augmented_data = AugmentedDataset("../traits_resized")
    data_loader = torch.utils.data.DataLoader(
                            augmented_data, batch_size=batch_size, shuffle=True
                        )
    img_shape = augmented_data[0].shape

    # initialize models
    noise_size = 100
    generator = Generator(img_shape, noise_size=noise_size)
    discriminator = Discriminator(img_shape)

    # set device of models
    generator.to(device)
    discriminator.to(device)

    # create optimizers (G=generator, D=discriminator)
    opt_G = optim.RMSprop(generator.parameters(), lr=0.00005)
    opt_D = optim.RMSprop(discriminator.parameters(), lr=0.00005)

    # train models for each epoch
    for epoch in range(epochs):
        
        # store losses as history
        discr_losses = []
        gen_losses = []
        
        # grab batch of real images
        for real_imgs in data_loader:
            
            # set device on images
            real_imgs = real_imgs.to(device)

            # store batch size
            batch_size = real_imgs.size(0)

            # - - - - - Train Discriminator - - - - -
            #      (train discr 10x more than gen)
            for _ in range(10):

                # reset gradient
                opt_D.zero_grad()

                # create fake images from noise (1 fake for each real)
                noise = torch.randn(batch_size, noise_size, 1, 1, device=device)
                fake_imgs_flat = generator(noise)
                fake_imgs = fake_imgs_flat.view(batch_size, *img_shape)

                # use discriminator to predict real and fake images
                #  (detach() to prevent this from updating gen's weight, only discr)
                y_real = discriminator(real_imgs)
                y_fake = discriminator(fake_imgs.detach())
                
                # get wasserstein loss, step down by gradient
                discr_loss = wass_disc_loss(y_real, y_fake, grad_penalty = True)
                discr_loss += 10*wass_disc_gp_loss(discriminator, real_imgs, fake_imgs, device=device)
                discr_loss.backward()
                opt_D.step()

                # record loss of this batch
                discr_losses.append(discr_loss.item())


            # - - - - - Train Generator - - - - -

            # reset gradient
            opt_G.zero_grad()

            # create new fakes to train on, have discr predict on them
            noise = torch.randn(batch_size, noise_size, 1, 1, device=device)
            fake_imgs_flat = generator(noise)
            fake_imgs = fake_imgs_flat.view(batch_size, *img_shape)
            y_fake = discriminator(fake_imgs)

            # get wasserstein loss, step down by gradient
            gen_loss = wass_gen_loss(y_fake)
            gen_loss.backward()
            opt_G.step()
            
            # record loss of this epoch
            gen_losses.append(gen_loss.item())

        if verbose:
            print(f"Epoch {epoch+1}:     Generator Loss {np.mean(gen_losses)}"
                  f"     Discriminator Loss {np.mean(discr_losses)}")

    # save models
    if save_gen_path is not None:
        generator.save(save_gen_path)

    if save_discr_path is not None:
        discriminator.save(save_discr_path)


if __name__ == '__main__':
    main_args = {
        'epochs': 10,
        'batch_size': 100,
        'save_gen_path': "../saved_models/gen.pkl",
        'save_discr_path': "../saved_models/discr.pkl",
        'verbose': True
    }
    train_Main(**main_args)


