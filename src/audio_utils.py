from pathlib import Path

import numpy as np
import soundfile as sf


REQUIRED_STEMS = [
    "mixture.wav",
    "vocals.wav",
    "drums.wav",
    "bass.wav",
    "other.wav",
]


def read_audio(path: Path):
    """
    Read stereo audio.

    Returns
    -------
    audio : np.ndarray
        Shape: [samples, channels]
    sr : int
        Sampling rate
    """

    audio, sr = sf.read(
        path,
        always_2d=True
    )

    return audio.astype(np.float64), sr


def load_track_segment(
    track_dir: Path,
    start_sec: float,
    duration_sec: float
):
    """
    Load a short MUSDB18-HQ segment.

    Returns mixture, vocals, accompaniment, sample rate.
    """

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    for filename in REQUIRED_STEMS:
        path = track_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

    # --------------------------------------------------------
    # Load stems
    # --------------------------------------------------------

    mixture, sr_mix = read_audio(
        track_dir / "mixture.wav"
    )

    vocals, sr_v = read_audio(
        track_dir / "vocals.wav"
    )

    drums, sr_d = read_audio(
        track_dir / "drums.wav"
    )

    bass, sr_b = read_audio(
        track_dir / "bass.wav"
    )

    other, sr_o = read_audio(
        track_dir / "other.wav"
    )

    sample_rates = {
        sr_mix,
        sr_v,
        sr_d,
        sr_b,
        sr_o,
    }

    if len(sample_rates) != 1:
        raise ValueError(
            "Stem sample rates do not match."
        )

    sr = sr_mix

    # --------------------------------------------------------
    # Construct accompaniment
    #
    # a[n] = drums + bass + other
    # --------------------------------------------------------

    accompaniment = drums + bass + other

    # --------------------------------------------------------
    # Extract segment
    # --------------------------------------------------------

    start = int(start_sec * sr)
    end = int(
        (start_sec + duration_sec) * sr
    )

    if end > len(mixture):
        raise ValueError(
            "Requested segment is longer than the track."
        )

    mixture = mixture[start:end]
    vocals = vocals[start:end]
    accompaniment = accompaniment[start:end]

    # --------------------------------------------------------
    # Verify:
    #
    # x[n] ≈ v[n] + a[n]
    # --------------------------------------------------------

    reconstructed_mix = vocals + accompaniment

    max_mix_error = np.max(
        np.abs(
            mixture - reconstructed_mix
        )
    )

    return {
        "mixture": mixture,
        "vocals": vocals,
        "accompaniment": accompaniment,
        "sr": sr,
        "mixture_consistency_error": max_mix_error,
    }