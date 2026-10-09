import numpy as np
import librosa

from config import (
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    WINDOW,
)


def multichannel_stft(audio):
    """
    Compute STFT separately for each channel.

    Parameters
    ----------
    audio : np.ndarray
        Shape [samples, channels]

    Returns
    -------
    spectrum : np.ndarray
        Shape [channels, frequency, time]
    """

    spectra = []

    for channel in range(
        audio.shape[1]
    ):

        spectrum = librosa.stft(
            audio[:, channel],
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=WINDOW,
            center=True,
        )

        spectra.append(spectrum)

    return np.stack(
        spectra,
        axis=0
    )


def multichannel_istft(
    spectrum,
    length
):
    """
    Reconstruct multichannel audio using ISTFT.

    Parameters
    ----------
    spectrum : np.ndarray
        [channels, frequency, time]

    length : int
        Target number of audio samples
    """

    channels = []

    for channel in range(
        spectrum.shape[0]
    ):

        audio = librosa.istft(
            spectrum[channel],
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=WINDOW,
            center=True,
            length=length,
        )

        channels.append(audio)

    return np.stack(
        channels,
        axis=1
    )