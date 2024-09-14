from PIL import Image
import os

def resize_images(directory, target_size=(400, 562)):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # Adjust the extension if necessary
            file_path = os.path.join(directory, filename)
            img = Image.open(file_path)
            img_resized = img.resize(target_size, Image.LANCZOS)
            img_resized.save(file_path)  # Overwrite the original image or save as new
            print(f"Resized {filename} to {target_size}")

        #break


resize_images('ages_images')  # Adjust the directory as needed