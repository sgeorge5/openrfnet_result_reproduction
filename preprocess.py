import torch
import numpy as np
from scipy.signal import stft
import torch.nn.functional as F


class STFTTransform:
    """
    Converts raw I/Q samples (2, T) into a normalized STFT spectrogram (1, H, W).
    This matches the preprocessing described in Paper 17.
    """

    def __init__(self, nperseg=64, noverlap=32, nfft=128, target_size=(224, 224)):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.target_size = target_size

    def __call__(self, iq_tensor):
        """
        iq_tensor: shape (2, T)
        returns: spectrogram tensor (1, H, W)
        """

        # Convert I/Q → complex signal
        i = iq_tensor[0].numpy()
        q = iq_tensor[1].numpy()
        complex_signal = i + 1j * q

        # Compute STFT
        f, t, Zxx = stft(
            complex_signal,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft
        )

        # Magnitude spectrogram
        S = np.abs(Zxx)

        # Convert to dB scale (Paper 17 uses 20*log10)
        S_db = 20 * np.log10(S + 1e-8)

        # Normalize to [0,1] (Paper 17 Eq. 4)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

        # Convert to torch tensor
        S_tensor = torch.tensor(S_norm, dtype=torch.float32)

        # Add channel dimension → (1, H, W)
        S_tensor = S_tensor.unsqueeze(0)

        # Resize to model input size
        S_tensor = F.interpolate(
            S_tensor.unsqueeze(0),
            size=self.target_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return S_tensor
