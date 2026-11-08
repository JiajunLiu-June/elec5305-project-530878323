"""Ground-truth-dependent diagnostics, not deployable blind separators."""
import numpy as np

def ideal_ratio_mask(vocal_stft, accompaniment_stft):
    pv, pa = np.abs(vocal_stft)**2, np.abs(accompaniment_stft)**2
    return np.divide(pv, pv+pa, out=np.zeros_like(pv), where=(pv+pa)>0)

def apply_irm(mixture_stft, vocal_stft, accompaniment_stft):
    mask = ideal_ratio_mask(vocal_stft, accompaniment_stft)
    return mask * mixture_stft, mask

def true_magnitude_mixture_phase(mixture_stft, vocal_stft):
    return np.abs(vocal_stft) * np.exp(1j*np.angle(mixture_stft))

def phase_sensitive_mask(mixture_stft, vocal_stft, upper=1.0):
    """Closest coefficient to V on the ray mX, restricted to 0 <= m <= upper.

    The coefficient-MSE optimum is Re(V conj(X))/|X|^2. Clipping imposes
    the ordinary attenuation-mask constraints. This does NOT imply a global
    waveform-SDR bound after overlapping ISTFT reconstruction.
    """
    power = np.abs(mixture_stft)**2
    mask = np.divide(np.real(vocal_stft*np.conj(mixture_stft)), power,
                     out=np.zeros_like(power), where=power>0)
    return np.clip(mask, 0, upper)

def oracle_estimates(X, V, A):
    irm, mask = apply_irm(X, V, A)
    return dict(irm=irm, magnitude_phase=true_magnitude_mixture_phase(X,V),
                clipped_psm=phase_sensitive_mask(X,V)*X,
                nonnegative_psm=phase_sensitive_mask(X,V,upper=np.inf)*X), mask
