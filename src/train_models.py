import torch
import torch.optim as optim
import numpy as np

from models import Generator, Discriminator

from util import load_dir_images, clip_weights
from util import wass_disc_loss, wass_gen_loss # wasserstein loss functions


'''
    Main file for training the discriminator and generator

'''

def train_Main(epochs=50, batch_size=100, 
               save_gen_path = None, save_discr_path = None, verbose=False):

    # check if cuda is available
    if (torch.cuda.is_available()):
        device = 'cuda'
    else:
        device = 'cpu'

    # load real data
    real_data = load_dir_images("../traits_resized")
    data_loader = torch.utils.data.DataLoader(
                            real_data, batch_size=batch_size, shuffle=True
                        )
    flat_img_size = real_data.size(1)

    # initialize models
    noise_size = 100
    generator = Generator(flat_img_size, noise_size=noise_size)
    discriminator = Discriminator(flat_img_size)

    # set device of models
    generator.to(device)
    discriminator.to(device)

    # create optimizers
    opt_G = optim.RMSprop(generator.parameters(), lr=0.00005)
    opt_D = optim.RMSprop(discriminator.parameters(), lr=0.00005)


    # iterate through epochs
    for epoch in range(epochs):

        discr_losses = []
        gen_losses = []
        
        # iterate through batches of real images 
        for real_imgs in data_loader:
            
            # set device on images
            real_imgs = real_imgs.to(device)

            # store batch size
            batch_size = real_imgs.size(0)


            # - - - - - Train Discriminator - - - - -
            for _ in range(10):
                # reset gradient
                opt_D.zero_grad()

                # create fake images from noise (1 fake for each real)
                noise = torch.randn(batch_size, noise_size, device=device)
                fake_imgs = generator(noise)

                # use discriminator to predict real and fake images
                #  (detach() to prevent this from updating generator's weights)
                y_real = discriminator(real_imgs)
                y_fake = discriminator(fake_imgs.detach())
                
                # get wasserstein loss, step down by gradient
                discr_loss = wass_disc_loss(y_real, y_fake)
                discr_loss.backward()
                opt_D.step()

                # clip weights
                clip_weights(discriminator, clip_value=0.01)
                
                discr_losses.append(discr_loss.item())

            # - - - - - Train Generator - - - - -

            # reset gradient
            opt_G.zero_grad()

            # create new fakes (since we detached above) to train on
            noise = torch.randn(batch_size, noise_size, device=device)
            fake_imgs = generator(noise)
            y_fake = discriminator(fake_imgs)

            # get wasserstein loss, step down by gradient
            gen_loss = wass_gen_loss(y_fake)
            gen_loss.backward()
            opt_G.step()

            gen_losses.append(gen_loss.item())

        if verbose:
            print(f"Epoch {epoch+1}:     Generator Loss {np.mean(gen_losses)}     Discriminator Loss {np.mean(discr_losses)}")

    # save models
    if save_gen_path is not None:
        generator.save(save_gen_path)

    if save_discr_path is not None:
        discriminator.save(save_discr_path)

if __name__ == '__main__':
    main_args = {
        'epochs': 1000,
        'batch_size': 100,
        'save_gen_path': "../saved_models/test_gen.pkl",
        'save_discr_path': "../saved_models/test_discr.pkl",
        'verbose': True
    }
    train_Main(**main_args)


