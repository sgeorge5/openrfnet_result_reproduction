import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class RFSignalDataset(Dataset):

    def __init__(self, data_dict, class_to_idx, transform=None):
        self.data_dict = data_dict
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.samples = []
        for class_name, arr in data_dict.items():
            for i in range(arr.shape[0]):
                self.samples.append((arr[i], class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iq, class_name = self.samples[idx]

        iq = torch.tensor(iq, dtype=torch.float32)

        if class_name in self.class_to_idx:
            label = self.class_to_idx[class_name]
        else:
            label = -1  

        if self.transform:
            iq = self.transform(iq)

        return iq, label, class_name


def load_pkl_dataset(path):
    return torch.load(path, weights_only=False)


def split_known_unknown_classes(all_classes, known_ratio=0.7, seed=42):
    random.seed(seed)
    classes = list(all_classes)
    random.shuffle(classes)

    k = int(len(classes) * known_ratio)
    known = classes[:k]
    unknown = classes[k:]

    return known, unknown


def build_datasets(data_dict, known_classes, unknown_classes):

    class_to_idx = {cls: i for i, cls in enumerate(known_classes)}
    train_dict = {cls: data_dict[cls] for cls in known_classes}
    test_dict = {cls: data_dict[cls] for cls in known_classes + unknown_classes}

    train_dataset = RFSignalDataset(train_dict, class_to_idx)
    test_dataset = RFSignalDataset(test_dict, class_to_idx)

    return train_dataset, test_dataset
