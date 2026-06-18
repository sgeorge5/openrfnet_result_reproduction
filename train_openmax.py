import torch
import numpy as np
from torch.utils.data import DataLoader
import os

from dataset import load_pkl_dataset, split_known_unknown_classes, build_datasets
from stft_transform import STFTTransform
from model import OpenRFNet
from openmax_utils import fit_weibull, extract_features

def main():
    DATA_PATH = "processed_raw_dataset.pt"
    MODEL_PATH = "checkpoints_supcon/best_model.pth"
    OPENMAX_PATH = "openmax_params.pt"
    BATCH_SIZE = 32
    KNOWN_RATIO = 0.7
    TAIL_SIZE = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dict = load_pkl_dataset(DATA_PATH)
    all_classes = list(data_dict.keys())

    known_classes, unknown_classes = split_known_unknown_classes(
        all_classes, known_ratio=KNOWN_RATIO
    )

    train_dataset, _ = build_datasets(
        data_dict, known_classes, unknown_classes
    )

    transform = STFTTransform()
    train_dataset.transform = transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = OpenRFNet(num_classes=len(known_classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Loaded closed-set model.")

    feats, labels = extract_features(model, train_loader, device)

    class_centers = {
        cls: feats[labels == cls].mean(axis=0)
        for cls in np.unique(labels)
    }

    weibull_models = fit_weibull(feats, labels, tail_size=TAIL_SIZE)
    print("Fitted Weibull models for OpenMax.")

    torch.save(
        {"centers": class_centers, "weibull": weibull_models},
        OPENMAX_PATH
    )
    print(f"Saved OpenMax parameters to {OPENMAX_PATH}")

if __name__ == "__main__":
    main()
