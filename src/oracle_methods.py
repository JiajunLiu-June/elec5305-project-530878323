import numpy as np

from config import EPS


def ideal_ratio_mask(
    vocal_stft,
    accompaniment_stft
):
    """
    Oracle Ideal Ratio Mask (IRM).

    M_IRM(f,t) =
        |V(f,t)|^2 /
        (|V(f,t)|^2 + |A(f,t)|^2 + epsilon)
    """

    vocal_power = (
        np.abs(vocal_stft) ** 2
    )

    accompaniment_power = (
        np.abs(accompaniment_stft) ** 2
    )

    mask = (
        vocal_power
        /
        (
            vocal_power
            + accompaniment_power
            + EPS
        )
    )

    return mask


def apply_irm(
    mixture_stft,
    vocal_stft,
    accompaniment_stft
):
    """
    Apply oracle IRM to mixture STFT.
    """

    mask = ideal_ratio_mask(
        vocal_stft,
        accompaniment_stft
    )

    vocal_estimate = (
        mask * mixture_stft
    )

    return vocal_estimate, mask


def true_magnitude_mixture_phase(
    mixture_stft,
    vocal_stft
):
    """
    Oracle reconstruction using:

    exact true vocal magnitude
    +
    mixture phase

    V_hat(f,t) =
        |V(f,t)| exp(j angle(X(f,t)))
    """

    magnitude = np.abs(
        vocal_stft
    )

    mixture_phase = np.angle(
        mixture_stft
    )

    estimate = (
        magnitude
        * np.exp(
            1j * mixture_phase
        )
    )

    return estimate