import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class IAMDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Check if 'lines.txt' exists
        lines_file = os.path.join(self.root_dir, 'ascii', 'lines.txt')
        if not os.path.exists(lines_file):
            raise FileNotFoundError(f"{lines_file} not found. Please download the IAM dataset.")
        
        self.data = self._load_labels(lines_file)

    def _load_labels(self, lines_file):
        # If you want to handle missing file gracefully, add custom data loading logic here
        with open(lines_file, 'r') as f:
            lines = f.readlines()

        # Return data (you can process this as needed)
        return [line.strip() for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Logic to get the image and label pair
        image_name = self.data[idx]  # Or load image path from some other source
        image = Image.open(os.path.join(self.root_dir, 'images', image_name))

        if self.transform:
            image = self.transform(image)

        return image, self.data[idx]
