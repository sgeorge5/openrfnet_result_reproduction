# openmax_utils.py
import numpy as np
from scipy.stats import weibull_min
from tqdm import tqdm
import torch

def fit_weibull(features, labels, tail_size=20):
    weibull_models = {}
    classes = np.unique(labels)

    for cls in classes:
        cls_features = features[labels == cls]
        center = cls_features.mean(axis=0)

        distances = np.linalg.norm(cls_features - center, axis=1)
        distances = np.sort(distances)[-tail_size:]

        shape, loc, scale = weibull_min.fit(distances, floc=0)
        weibull_models[int(cls)] = (shape, scale)

    return weibull_models


def openmax_score(feature, class_centers, weibull_models):
    distances = {
        cls: np.linalg.norm(feature - class_centers[cls])
        for cls in class_centers
    }

    unknown_scores = []
    for cls, dist in distances.items():
        shape, scale = weibull_models[cls]
        wscore = weibull_min.cdf(dist, shape, scale=scale)
        unknown_scores.append(wscore)

    return max(unknown_scores)


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
