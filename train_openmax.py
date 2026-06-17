import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import weibull_min
import os
from tqdm import tqdm

from dataset import load_pkl_dataset, split_known_unknown_classes, build_datasets
from preprocess import STFTTransform
from model import OpenRFNet


# ---------------------------------------------------------
# 1. Fit Weibull distributions for OpenMax
# ---------------------------------------------------------
def fit_weibull(features, labels, tail_size=20):
    """
    Fits a Weibull distribution to each class's feature distances.
    Returns a dict: {class_idx: (shape, scale)}
    """

    weibull_models = {}
    classes = np.unique(labels)

    for cls in classes:
        cls_features = features[labels == cls]
        center = cls_features.mean(axis=0)

        # Compute distances to class center
        distances = np.linalg.norm(cls_features - center, axis=1)
        distances = np.sort(distances)[-tail_size:]  # tail samples

        # Fit Weibull
        shape, loc, scale = weibull_min.fit(distances, floc=0)
        weibull_models[int(cls)] = (shape, scale)

    return weibull_models


# ---------------------------------------------------------
# 2. OpenMax scoring
# ---------------------------------------------------------
def openmax_score(feature, class_centers, weibull_models):
    """
    Computes OpenMax probability of being unknown.
    """

    distances = {
        cls: np.linalg.norm(feature - class_centers[cls])
        for cls in class_centers
    }

    # Compute Weibull CDF for each class
    unknown_scores = []
    for cls, dist in distances.items():
        shape, scale = weibull_models[cls]
        wscore = weibull_min.cdf(dist, shape, scale=scale)
        unknown_scores.append(wscore)

    # Unknown probability = max Weibull score
    return max(unknown_scores)


# ---------------------------------------------------------
# 3. Extract features from model
# ---------------------------------------------------------
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []

    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Extracting features"):
            x = x.to(device)
            fused_feat, _, _ = model(x, return_features=True)
            feats.append(fused_feat.cpu().numpy())
            labels.append(y.numpy())

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels


# ---------------------------------------------------------
# 4. Main OpenMax training + evaluation
# ---------------------------------------------------------
def main():
    # -----------------------------
    # Config
    # -----------------------------
    DATA_PATH = "data/your_dataset.pkl"
    MODEL_PATH = "checkpoints_supcon/best_model.pth"
    BATCH_SIZE = 32
    KNOWN_RATIO = 0.7
    TAIL_SIZE = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load dataset
    # -----------------------------
    data_dict = load_pkl_dataset(DATA_PATH)
    all_classes = list(data_dict.keys())

    known_classes, unknown_classes = split_known_unknown_classes(
        all_classes, known_ratio=KNOWN_RATIO
    )

    train_dataset, test_dataset = build_datasets(
        data_dict, known_classes, unknown_classes
    )

    transform = STFTTransform()
    train_dataset.transform = transform
    test_dataset.transform = transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------
    # Load trained model
    # -----------------------------
    model = OpenRFNet(num_classes=len(known_classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Loaded closed-set model.")

    # -----------------------------
    # Extract features for known classes
    # -----------------------------
    feats, labels = extract_features(model, train_loader, device)

    # Compute class centers
    class_centers = {
        cls: feats[labels == cls].mean(axis=0)
        for cls in np.unique(labels)
    }

    # Fit Weibull models
    weibull_models = fit_weibull(feats, labels, tail_size=TAIL_SIZE)
    print("Fitted Weibull models for OpenMax.")

    # -----------------------------
    # Evaluate OpenMax
    # -----------------------------
    correct_known = 0
    correct_unknown = 0
    total_known = 0
    total_unknown = 0

    with torch.no_grad():
        for x, y, class_names in tqdm(test_loader, desc="Evaluating OpenMax"):
            x = x.to(device)
            fused_feat, _, logits = model(x, return_features=True)

            fused_feat = fused_feat.cpu().numpy()
            logits = logits.cpu().numpy()

            for i in range(len(y)):
                feature = fused_feat[i]
                true_label = y[i].item()
                class_name = class_names[i]

                # Compute unknown probability
                unk_prob = openmax_score(feature, class_centers, weibull_models)

                if class_name in known_classes:
                    total_known += 1
                    pred = np.argmax(logits[i])
                    if pred == true_label and unk_prob < 0.5:
                        correct_known += 1
                else:
                    total_unknown += 1
                    if unk_prob >= 0.5:
                        correct_unknown += 1

    # -----------------------------
    # Results
    # -----------------------------
    known_acc = correct_known / total_known
    unknown_acc = correct_unknown / total_unknown
    open_set_acc = (correct_known + correct_unknown) / (total_known + total_unknown)

    print("\n===== OpenMax Results =====")
    print(f"Known-class accuracy:   {known_acc:.4f}")
    print(f"Unknown detection rate: {unknown_acc:.4f}")
    print(f"Open-set accuracy:      {open_set_acc:.4f}")


if __name__ == "__main__":
    main()
