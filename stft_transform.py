# stft_transform.py
import torch
import numpy as np
from scipy.signal import stft
import torch.nn.functional as F

class STFTTransform:
    def __init__(self, nperseg=64, noverlap=32, nfft=128, target_size=(224, 224)):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.target_size = target_size

    def __call__(self, iq_tensor):
        i = iq_tensor[0].numpy()
        q = iq_tensor[1].numpy()
        complex_signal = i + 1j * q

        f, t, Zxx = stft(
            complex_signal,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft
        )

        S = np.abs(Zxx)
        S_db = 20 * np.log10(S + 1e-8)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

        S_tensor = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)
        S_tensor = F.interpolate(
            S_tensor.unsqueeze(0),
            size=self.target_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return S_tensor
