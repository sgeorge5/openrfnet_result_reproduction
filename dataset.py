import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random


class RFSignalDataset(Dataset):
    """
    Loads I/Q samples from a .pkl dictionary and returns:
        - raw I/Q tensor (2, T)
        - label index
        - class name
    STFT conversion happens later in preprocess.py
    """

    def __init__(self, data_dict, class_to_idx, transform=None):
        self.data_dict = data_dict
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.samples = []
        for class_name, arr in data_dict.items():
            label = class_to_idx[class_name]
            for i in range(arr.shape[0]):
                self.samples.append((arr[i], label, class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iq, label, class_name = self.samples[idx]

        # Convert to tensor
        iq = torch.tensor(iq, dtype=torch.float32)

        # Optional transform (e.g., STFT)
        if self.transform:
            iq = self.transform(iq)

        return iq, label, class_name


def load_pkl_dataset(path):
    """
    Loads the .pkl file and returns a Python dictionary:
        { 'QPSK': np.array([...]), 'QAM16': np.array([...]), ... }
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def split_known_unknown_classes(all_classes, known_ratio=0.7, seed=42):
    """
    Randomly splits classes into known and unknown sets.
    Example:
        known_classes = ['QPSK', 'QAM16', ...]
        unknown_classes = ['GFSK', 'AM', ...]
    """
    random.seed(seed)
    classes = list(all_classes)
    random.shuffle(classes)

    k = int(len(classes) * known_ratio)
    known = classes[:k]
    unknown = classes[k:]

    return known, unknown


def build_datasets(data_dict, known_classes, unknown_classes):
    """
    Builds PyTorch datasets for:
        - closed-set training (known classes only)
        - open-set testing (known + unknown)
    """

    # Map class names to integer labels
    class_to_idx = {cls: i for i, cls in enumerate(known_classes)}

    # Closed-set training data
    train_dict = {cls: data_dict[cls] for cls in known_classes}

    # Open-set test data (known + unknown)
    test_dict = {cls: data_dict[cls] for cls in known_classes + unknown_classes}

    train_dataset = RFSignalDataset(train_dict, class_to_idx)
    test_dataset = RFSignalDataset(test_dict, class_to_idx)

    return train_dataset, test_dataset
