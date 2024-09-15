import torch
from PIL import Image
import numpy as np
import os

from models import Generator

''' File for generating images using generator '''


def generate_images_main(num_images, save_path, gen_model_path = "../saved_models/gen.pkl", 
                         noise_size=100, ex_realimg_path = "../traits_resized/acrobatic.png"):


    # open example real img to get flat image size and dimensions
    dir_path = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(dir_path, ex_realimg_path)
    ex_img = Image.open(img_path)
    img_width, img_height = ex_img.size
    flat_img_size = img_width*img_height*3 # 3 for rgb


    # load model
    generator = Generator(flat_img_size, noise_size=noise_size)
    generator.load(gen_model_path)

    # generate images
    noise = torch.randn(num_images, noise_size)
    gen_imgs = generator(noise)

    # reshape images to go back to PIL
    save_dir = os.path.join(dir_path, save_path)
    for i, img_tensor in enumerate(gen_imgs):

        img_reshaped = img_tensor.view(img_height, img_width, 3)
        img_np = img_reshaped.detach().cpu().numpy()
        img_np = (img_np*255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        img_pil.save(f"{save_dir}/gen_img_{i}.png")


if __name__ == '__main__':
    main_args = {
        'num_images': 5,
        'save_path': "../generated_imgs"
    }
    generate_images_main(**main_args)