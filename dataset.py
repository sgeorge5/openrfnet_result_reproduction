import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class RFSignalDataset(Dataset):
    """
    Loads I/Q samples from a dict and returns:
        - raw I/Q tensor (2, T)
        - label index (known classes only)
        - class name (string)
    """

    def __init__(self, data_dict, class_to_idx, transform=None):
        self.data_dict = data_dict
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.samples = []
        for class_name, arr in data_dict.items():
            for i in range(arr.shape[0]):
                # Store only IQ and class_name — label assigned later
                self.samples.append((arr[i], class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iq, class_name = self.samples[idx]

        iq = torch.tensor(iq, dtype=torch.float32)

        # Assign label only for known classes
        if class_name in self.class_to_idx:
            label = self.class_to_idx[class_name]
        else:
            label = -1  # unknown class

        if self.transform:
            iq = self.transform(iq)

        return iq, label, class_name


# ---------------------------------------------------------
# Load processed dataset
# ---------------------------------------------------------
def load_pkl_dataset(path):
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------
# Split classes into known/unknown
# ---------------------------------------------------------
def split_known_unknown_classes(all_classes, known_ratio=0.7, seed=42):
    random.seed(seed)
    classes = list(all_classes)
    random.shuffle(classes)

    k = int(len(classes) * known_ratio)
    known = classes[:k]
    unknown = classes[k:]

    return known, unknown


# ---------------------------------------------------------
# Build train/test datasets
# ---------------------------------------------------------
def build_datasets(data_dict, known_classes, unknown_classes):
    # Only known classes get labels
    class_to_idx = {cls: i for i, cls in enumerate(known_classes)}

    # Closed-set training uses only known classes
    train_dict = {cls: data_dict[cls] for cls in known_classes}

    # Open-set testing uses known + unknown
    test_dict = {cls: data_dict[cls] for cls in known_classes + unknown_classes}

    train_dataset = RFSignalDataset(train_dict, class_to_idx)
    test_dataset = RFSignalDataset(test_dict, class_to_idx)

    return train_dataset, test_dataset
