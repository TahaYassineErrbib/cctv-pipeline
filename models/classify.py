"""
Shared classification helper used by C1, C2, C3, and C4.
Keeping this in one place avoids re-introducing the PIL/tensor conversion
bug we already hit once in the original monolithic pipeline.py.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import config
from models.loaders import DEVICE


eval_transform = transforms.Compose([
    transforms.Resize((config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])


@torch.no_grad()
def classify(model, classes, crop_bgr):
    """
    Runs a classifier on a BGR crop (as given by cv2/YOLO).
    Returns (label, confidence). Returns (None, 0.0) for empty/invalid crops.

    NOTE: eval_transform's Resize/ToTensor steps require a PIL Image —
    NOT a raw ndarray or a pre-built tensor. Always convert via
    PIL.Image.fromarray() before calling eval_transform.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    tensor = eval_transform(pil_img).unsqueeze(0).to(DEVICE)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])
