from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os

class WatermarkDataset(Dataset):
    def __init__(self, root_dir):
        self.watermarked_dir = os.path.join(root_dir, "watermarked")
        self.clean_dir = os.path.join(root_dir, "clean")
        self.watermarked_images = os.listdir(self.watermarked_dir)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.watermarked_images)

    def __getitem__(self, idx):
        watermarked_filename = self.watermarked_images[idx]
        watermarked_path = os.path.join(self.watermarked_dir, watermarked_filename)

        clean_filename = os.path.splitext(watermarked_filename)[0]
        clean_path_jpg = os.path.join(self.clean_dir, clean_filename + ".jpg")
        clean_path_png = os.path.join(self.clean_dir, clean_filename + ".png")

        if os.path.exists(clean_path_jpg):
            clean_path = clean_path_jpg
        elif os.path.exists(clean_path_png):
            clean_path = clean_path_png
        else:
            raise FileNotFoundError(f"No matching file found for {clean_filename} in clean directory")

        watermarked_image = Image.open(watermarked_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")

        watermarked_image = self.transform(watermarked_image)
        clean_image = self.transform(clean_image)

        return watermarked_image, clean_image
