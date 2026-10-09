import numpy as np

from config import EPS


def vocal_to_accompaniment_ratio(
    vocals,
    accompaniment
):
    """
    Vocal-to-Accompaniment Ratio (VAR)

    VAR =
        10 log10(
            sum(v[n]^2)
            /
            sum(a[n]^2)
        )
    """

    vocal_energy = np.sum(
        vocals ** 2
    )

    accompaniment_energy = np.sum(
        accompaniment ** 2
    )

    var = (
        10
        * np.log10(
            (vocal_energy + EPS)
            /
            (accompaniment_energy + EPS)
        )
    )

    return var


def tf_overlap(
    vocal_stft,
    accompaniment_stft
):
    """
    Time-frequency overlap.

    O =
        sum min(|V|^2, |A|^2)
        /
        (sum |V|^2 + epsilon)
    """

    vocal_power = (
        np.abs(vocal_stft) ** 2
    )

    accompaniment_power = (
        np.abs(accompaniment_stft) ** 2
    )

    overlap_energy = np.minimum(
        vocal_power,
        accompaniment_power
    )

    overlap = (
        np.sum(overlap_energy)
        /
        (
            np.sum(vocal_power)
            + EPS
        )
    )

    return overlap


def phase_disagreement(
    vocal_stft,
    mixture_stft
):
    """
    Energy-weighted phase disagreement.

    D_phi =
        sum |V|^2 [1 - cos(delta_phi)]
        /
        (sum |V|^2 + epsilon)

    where

    delta_phi =
        angle(V) - angle(X)
    """

    vocal_power = (
        np.abs(vocal_stft) ** 2
    )

    delta_phi = (
        np.angle(vocal_stft)
        - np.angle(mixture_stft)
    )

    numerator = np.sum(
        vocal_power
        * (
            1
            - np.cos(delta_phi)
        )
    )

    denominator = (
        np.sum(vocal_power)
        + EPS
    )

    return numerator / denominator


def simple_sdr(
    reference,
    estimate
):
    """
    Preliminary SDR-style sanity check.

    IMPORTANT:
    This is not the final project evaluation.
    Final evaluation will use museval.
    """

    error = (
        reference - estimate
    )

    reference_energy = np.sum(
        reference ** 2
    )

    error_energy = np.sum(
        error ** 2
    )

    return (
        10
        * np.log10(
            (reference_energy + EPS)
            /
            (error_energy + EPS)
        )
    )