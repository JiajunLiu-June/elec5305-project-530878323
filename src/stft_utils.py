"""Shared analysis STFT. Learned models retain their own native transforms."""
import numpy as np
import librosa
from .config import N_FFT, HOP_LENGTH

def multichannel_stft(audio, n_fft=N_FFT, hop_length=None):
    audio = np.asarray(audio)
    if audio.ndim != 2 or not audio.size or not np.isfinite(audio).all():
        raise ValueError('Audio must be finite [samples, channels]')
    hop = hop_length or (HOP_LENGTH if n_fft == N_FFT else n_fft // 4)
    return librosa.stft(audio.T, n_fft=n_fft, win_length=n_fft,
                        hop_length=hop, window='hann', center=True, pad_mode='constant')

def multichannel_istft(spectrum, length, n_fft=N_FFT, hop_length=None):
    hop = hop_length or (HOP_LENGTH if n_fft == N_FFT else n_fft // 4)
    return librosa.istft(spectrum, n_fft=n_fft, win_length=n_fft,
                         hop_length=hop, window='hann', center=True, length=length).T
