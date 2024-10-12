import torch
from PIL import Image
from torchvision import transforms
import os

from models import Generator
from util import adjust_contrast_based_on_brightness, normalize_noise

''' File for generating images using generator '''


def generate_images_main(num_images, save_path, 
                        gen_model_path = "../saved_models/gen.pkl",
                        noise_size=128, 
                        ex_realimg_path = "../traits_resized/acrobatic.png"):
    ''' 
        Main function for generating and saving images using a pretrained model

        Arguments:
        - num_images: number of images to generate
        - save_path: path to save the generated images
        - gen_model_path: path to the pretrained generator model 
        - noise_size: size of the noise vector
        - ex_realimg_path: path to an ex of an image that model was trained 
                           on for dimensionality purposes
    '''


    # open example real img to get shape
    ex_realimg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ex_realimg_path)
    ex_img = Image.open(ex_realimg_path)
    img_width, img_height = ex_img.size
    img_shape = [3, img_height, img_width]

    # load model
    generator = Generator(img_shape, noise_size=noise_size)
    generator.load(gen_model_path)

    # generate images
    noise = torch.randn(num_images, noise_size, 1, 1)
    noise = normalize_noise(noise)
    gen_imgs = generator(noise)

    # reshape images to go back to PIL
    tens_to_pil = transforms.ToPILImage()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_path)
    for i, img_tensor in enumerate(gen_imgs):

        # reshape
        img_t = img_tensor.reshape(*img_shape)

        # convert to pil and save
        img_pil = tens_to_pil(img_t)
        img_pil = adjust_contrast_based_on_brightness(img_pil)

        # resize image so easier to see
        img_pil = img_pil.resize((img_pil.size[0]*2, img_pil.size[1]*2))
        img_pil.save(f"{save_path}/img{i}.png")


if __name__ == '__main__':

    gen_images_args = {
        'num_images': 20,
        'save_path': "../generated_imgs"
    }
    generate_images_main(**gen_images_args)