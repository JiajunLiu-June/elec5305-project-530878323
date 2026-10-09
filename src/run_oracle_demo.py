import argparse
from pathlib import Path

import soundfile as sf

from config import (
    DATASET_ROOT,
    RESULTS_ROOT,
    DEFAULT_START_SEC,
    DEFAULT_DURATION_SEC,
)

from audio_utils import (
    load_track_segment,
)

from stft_utils import (
    multichannel_stft,
    multichannel_istft,
)

from oracle_methods import (
    apply_irm,
    true_magnitude_mixture_phase,
)

from metrics import (
    vocal_to_accompaniment_ratio,
    tf_overlap,
    phase_disagreement,
    simple_sdr,
)

from plotting import (
    save_spectrogram,
    save_mask,
)


def find_track(track_name=None):

    test_root = (
        DATASET_ROOT / "test"
    )

    if not test_root.exists():
        raise FileNotFoundError(
            f"""
MUSDB18-HQ test directory not found:

{test_root}

Put the dataset in:
data/MUSDB18-HQ/
"""
        )

    tracks = sorted(
        [
            p
            for p in test_root.iterdir()
            if p.is_dir()
        ]
    )

    if len(tracks) == 0:
        raise RuntimeError(
            "No MUSDB18-HQ test tracks found."
        )

    if track_name is None:
        return tracks[0]

    track = (
        test_root / track_name
    )

    if not track.exists():
        raise FileNotFoundError(
            f"Track not found: {track}"
        )

    return track


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="MUSDB18-HQ test track folder name",
    )

    parser.add_argument(
        "--start",
        type=float,
        default=DEFAULT_START_SEC,
        help="Segment start time in seconds",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SEC,
        help="Segment duration in seconds",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Select track
    # --------------------------------------------------------

    track_dir = find_track(
        args.track
    )

    track_name = track_dir.name

    print()
    print(
        f"Track: {track_name}"
    )

    print(
        f"Segment: {args.start:.1f} - "
        f"{args.start + args.duration:.1f} s"
    )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    data = load_track_segment(
        track_dir,
        args.start,
        args.duration,
    )

    mixture = data["mixture"]
    vocals = data["vocals"]
    accompaniment = data[
        "accompaniment"
    ]

    sr = data["sr"]

    print(
        f"Sample rate: {sr} Hz"
    )

    print(
        "Mixture consistency error:",
        data[
            "mixture_consistency_error"
        ],
    )

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    X = multichannel_stft(
        mixture
    )

    V = multichannel_stft(
        vocals
    )

    A = multichannel_stft(
        accompaniment
    )

    # --------------------------------------------------------
    # Oracle 1: IRM
    # --------------------------------------------------------

    V_irm, irm = apply_irm(
        X,
        V,
        A,
    )

    vocal_irm = (
        multichannel_istft(
            V_irm,
            length=len(vocals),
        )
    )

    # --------------------------------------------------------
    # Oracle 2:
    # True vocal magnitude + mixture phase
    # --------------------------------------------------------

    V_mag_phase = (
        true_magnitude_mixture_phase(
            X,
            V,
        )
    )

    vocal_mag_phase = (
        multichannel_istft(
            V_mag_phase,
            length=len(vocals),
        )
    )

    # --------------------------------------------------------
    # Acoustic difficulty variables
    # --------------------------------------------------------

    var = (
        vocal_to_accompaniment_ratio(
            vocals,
            accompaniment,
        )
    )

    overlap = tf_overlap(
        V,
        A,
    )

    phase = phase_disagreement(
        V,
        X,
    )

    # --------------------------------------------------------
    # Preliminary SDR sanity checks
    # --------------------------------------------------------

    irm_sdr = simple_sdr(
        vocals,
        vocal_irm,
    )

    mag_phase_sdr = simple_sdr(
        vocals,
        vocal_mag_phase,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    safe_track_name = (
        track_name
        .replace("/", "_")
    )

    output_dir = (
        RESULTS_ROOT
        / "first_example"
        / safe_track_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save audio
    # --------------------------------------------------------

    sf.write(
        output_dir / "mixture.wav",
        mixture,
        sr,
    )

    sf.write(
        output_dir / "true_vocal.wav",
        vocals,
        sr,
    )

    sf.write(
        output_dir / "accompaniment.wav",
        accompaniment,
        sr,
    )

    sf.write(
        output_dir / "irm_vocal.wav",
        vocal_irm,
        sr,
    )

    sf.write(
        output_dir
        / "true_magnitude_mixture_phase.wav",
        vocal_mag_phase,
        sr,
    )

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------

    save_spectrogram(
        X,
        sr,
        "Mixture Spectrogram",
        output_dir
        / "01_mixture_spectrogram.png",
    )

    save_spectrogram(
        V,
        sr,
        "True Vocal Spectrogram",
        output_dir
        / "02_true_vocal_spectrogram.png",
    )

    save_spectrogram(
        A,
        sr,
        "Accompaniment Spectrogram",
        output_dir
        / "03_accompaniment_spectrogram.png",
    )

    save_mask(
        irm,
        sr,
        "Ideal Ratio Mask",
        output_dir
        / "04_ideal_ratio_mask.png",
    )

    save_spectrogram(
        V_irm,
        sr,
        "IRM Vocal Estimate",
        output_dir
        / "05_irm_vocal_spectrogram.png",
    )

    save_spectrogram(
        V_mag_phase,
        sr,
        "True Magnitude + Mixture Phase",
        output_dir
        / "06_true_mag_mix_phase.png",
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_text = f"""
ELEC5305 Oracle Experiment
==========================

Track:
{track_name}

Segment:
{args.start:.2f} - {args.start + args.duration:.2f} s

Sample rate:
{sr} Hz

Acoustic difficulty
-------------------

VAR:
{var:.4f} dB

Time-frequency overlap O:
{overlap:.6f}

Phase disagreement D_phi:
{phase:.6f}

Preliminary reconstruction
--------------------------

IRM SDR-style score:
{irm_sdr:.4f} dB

True magnitude + mixture phase:
{mag_phase_sdr:.4f} dB

NOTE:
These SDR values are preliminary sanity checks.
Final source-separation evaluation will use museval.
"""

    with open(
        output_dir / "metrics.txt",
        "w",
    ) as file:

        file.write(
            metrics_text.strip()
        )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print()
    print(
        "========== RESULTS =========="
    )

    print(
        f"VAR: {var:.3f} dB"
    )

    print(
        f"TF overlap: {overlap:.4f}"
    )

    print(
        f"Phase disagreement: "
        f"{phase:.4f}"
    )

    print()
    print(
        f"IRM preliminary SDR: "
        f"{irm_sdr:.3f} dB"
    )

    print(
        "True magnitude + mixture "
        f"phase SDR: {mag_phase_sdr:.3f} dB"
    )

    print()
    print(
        f"Saved to:\n{output_dir}"
    )


if __name__ == "__main__":
    main()