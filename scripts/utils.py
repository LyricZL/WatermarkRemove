# scripts/utils.py
from PIL import Image
import torch
import numpy as np

def load_image(image_path, size=(256, 256)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(size)
    image = torch.tensor(np.array(image).transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0)
    return image
