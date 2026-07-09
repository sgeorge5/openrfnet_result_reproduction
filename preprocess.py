import pickle
import torch
import numpy as np
from collections import defaultdict

MAX_SAMPLES_PER_CLASS = 500


def main():
    print("=== Preprocessing Started ===")

    with open("RML2016data.pkl", "rb") as f:
        raw_data = pickle.load(f, encoding="latin1")

    merged = defaultdict(list)

    for (mod, snr), samples in raw_data.items():
        merged[mod].append(samples)

    merged = {mod: torch.tensor(np.concatenate(blocks, axis=0)) 
              for mod, blocks in merged.items()}

    reduced = {}
    for mod, arr in merged.items():
        reduced[mod] = arr[:MAX_SAMPLES_PER_CLASS]

    torch.save(reduced, "processed_raw_dataset.pt")

    print("=== Preprocessing Complete ===")
    print(f"Saved reduced dataset with {MAX_SAMPLES_PER_CLASS} samples per class.")

if __name__ == "__main__":
    main()
