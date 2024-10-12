from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

''' Class for applying augmentations on dataset on __get___'''
class AugmentedDataset(Dataset):

    def __init__(self, image_folder, resize=(144,104)):

        # get paths to images
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_folder = os.path.join(curr_dir, image_folder)
        self.image_paths = [os.path.join(self.image_folder, img) for img in os.listdir(self.image_folder)]

        # Define transformations and augmentations
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.augmentations = [
            transforms.Compose([]),  # No transformation (original image)
            transforms.Compose([transforms.ColorJitter()])  # Color jitter
        ]

    def __len__(self):
        # Lenght of dataset is multipled by number of augmentations
        return len(self.image_paths) * (len(self.augmentations) if self.augmentations else 1)

    def __getitem__(self, idx):
        ''' apply augmentations when an item is retrieved '''

        # get the image_idx and aug_idx based on the index of the full augmented set
        image_idx = idx // (len(self.augmentations) if self.augmentations else 1)
        aug_idx = idx % (len(self.augmentations) if self.augmentations else 1)

        # Load the image
        image_path = self.image_paths[image_idx]
        image = Image.open(image_path).convert('RGB')

        # Apply augmentations
        if self.augmentations:
            image = self.augmentations[aug_idx](image)

        # Apply additional transformations (like converting to tensor)
        if self.transform:
            image = self.transform(image)

        return image
