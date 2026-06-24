"""
Loads all models used by the pipeline: YOLOv8 detector + the four
hierarchical garment classifiers (C1, C2, C3, C4).
"""

import os

import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

import config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yolo_detector():
    """Loads the YOLOv8 model used for person detection + tracking."""
    model = YOLO(config.YOLO_WEIGHTS)
    print(f"[YOLO] Loaded {config.YOLO_WEIGHTS} | device={DEVICE}")
    return model


def load_resnet50_classifier(ckpt_path):
    """
    Loads a ResNet-50 classifier saved by 02_train_C2_C3_C4_combined.ipynb,
    which stores: {'model_state_dict': ..., 'classes': [...], 'best_val_f1': ...}
    Used for C2, C3, C4.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    state_dict = ckpt["model_state_dict"]

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE).eval()

    print(f"[Classifier] Loaded {os.path.basename(ckpt_path)} | classes={classes}")
    return model, classes


def load_c1_garment_type():
    """
    Loads the pre-existing C1 model (garment_type_v2.pth).
    Architecture: ResNet-18 with fc = Sequential(Dropout, Linear).
    Confirmed via runtime logs: classes = ['long', 'standard'].
    """
    ckpt = torch.load(config.C1_CKPT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    classes = ckpt.get("classes") if isinstance(ckpt, dict) else None
    if classes is None:
        classes = config.C1_CLASSES_FALLBACK
        print(f"[C1] No class list found in checkpoint, using fallback: {classes}")

    num_classes = len(classes)

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE).eval()

    print(f"[C1] Loaded {os.path.basename(config.C1_CKPT_PATH)} | classes={classes}")
    return model, classes


def load_all_models():
    """
    Convenience loader: returns everything main.py needs in one call.
    Returns a dict so callers can access models by name instead of
    juggling a long positional tuple.
    """
    print("Loading all models...")
    yolo_model = load_yolo_detector()
    c1_model, c1_classes = load_c1_garment_type()
    c2_model, c2_classes = load_resnet50_classifier(config.C2_CKPT_PATH)
    c3_model, c3_classes = load_resnet50_classifier(config.C3_CKPT_PATH)
    c4_model, c4_classes = load_resnet50_classifier(config.C4_CKPT_PATH)
    print("All models loaded.\n")

    return {
        "yolo": yolo_model,
        "c1": (c1_model, c1_classes),
        "c2": (c2_model, c2_classes),
        "c3": (c3_model, c3_classes),
        "c4": (c4_model, c4_classes),
    }
