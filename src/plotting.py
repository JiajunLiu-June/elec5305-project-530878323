from pathlib import Path

import numpy as np
import librosa
import librosa.display

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config import HOP_LENGTH


def save_spectrogram(
    spectrum,
    sr,
    title,
    output_path: Path
):
    """
    Save stereo-averaged magnitude spectrogram.
    """

    magnitude = np.mean(
        np.abs(spectrum),
        axis=0
    )

    db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max
    )

    plt.figure(
        figsize=(10, 5)
    )

    librosa.display.specshow(
        db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="log",
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


def save_mask(
    mask,
    sr,
    title,
    output_path: Path
):
    """
    Save Ideal Ratio Mask visualisation.
    """

    mask_display = np.mean(
        mask,
        axis=0
    )

    plt.figure(
        figsize=(10, 5)
    )

    librosa.display.specshow(
        mask_display,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="log",
    )

    plt.colorbar()

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()