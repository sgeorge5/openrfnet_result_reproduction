import os
import torch
import numpy as np
import random


# ---------------------------------------------------------
# Set random seeds for reproducibility

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Ensure directory exists
# ---------------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ---------------------------------------------------------
# Save model checkpoint
# ---------------------------------------------------------
def save_checkpoint(model, path):
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")


# ---------------------------------------------------------
# Load model checkpoint
# ---------------------------------------------------------
def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Loaded checkpoint: {path}")
    return model


# ---------------------------------------------------------
# Extract features from a model (used in OpenMax)
# ---------------------------------------------------------
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            fused_feat, _, _ = model(x, return_features=True)

            feats.append(fused_feat.cpu().numpy())
            labels.append(y.numpy())

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels


# ---------------------------------------------------------
# Simple logger
# ---------------------------------------------------------
class Logger:
    def __init__(self, filepath):
        ensure_dir(os.path.dirname(filepath))
        self.filepath = filepath

    def log(self, text):
        with open(self.filepath, "a") as f:
            f.write(text + "\n")
        print(text)
