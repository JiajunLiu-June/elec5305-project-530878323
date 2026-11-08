"""Explicit metrics: reconstruction SDR is not BSS Eval SDR or SI-SDR."""
import numpy as np

TINY = np.finfo(np.float64).tiny

def _energy(x):
    return float(np.sum(np.abs(x)**2, dtype=np.float64))

def _ratio_db(a,b):
    return float(10*(np.log10(max(a,TINY))-np.log10(max(b,TINY))))

def vocal_to_accompaniment_ratio(vocals, accompaniment):
    if _energy(vocals) <= 0 or _energy(accompaniment) <= 0:
        raise ValueError('VAR requires nonzero vocal and accompaniment energy')
    return _ratio_db(_energy(vocals), _energy(accompaniment))

def tf_overlap(vocal_stft, accompaniment_stft):
    v, a = np.abs(vocal_stft)**2, np.abs(accompaniment_stft)**2
    if np.sum(v) <= 0:
        raise ValueError('Overlap is undefined for silent vocals')
    return float(np.sum(np.minimum(v,a))/np.sum(v))

def phase_disagreement(vocal_stft, mixture_stft):
    power = np.abs(vocal_stft)**2
    if np.sum(power) <= 0:
        raise ValueError('Phase disagreement is undefined for silent vocals')
    delta = np.angle(vocal_stft)-np.angle(mixture_stft)
    return float(np.sum(power*(1-np.cos(delta)))/np.sum(power))

def reconstruction_sdr(reference, estimate):
    """10 log10(sum ref^2 / sum (ref-est)^2), no fitted gain/filter/delay."""
    reference, estimate = np.asarray(reference), np.asarray(estimate)
    if reference.shape != estimate.shape or _energy(reference) <= 0:
        raise ValueError('Metric requires equal shapes and an active reference')
    return _ratio_db(_energy(reference), _energy(reference-estimate))

# Compatibility name retained for the original script, now accurately documented.
simple_sdr = reconstruction_sdr

def si_sdr(reference, estimate):
    """Concatenate stereo after per-channel demeaning; one shared fitted scale."""
    r, e = np.asarray(reference), np.asarray(estimate)
    if r.shape != e.shape:
        raise ValueError('Reference and estimate shapes differ')
    r = (r-r.mean(axis=0)).ravel()
    e = (e-e.mean(axis=0)).ravel()
    if _energy(r) <= 0:
        raise ValueError('SI-SDR requires nonconstant reference audio')
    target = np.dot(r,e)/np.dot(r,r)*r
    return _ratio_db(_energy(target), _energy(e-target))

def score(reference, estimate):
    return dict(reconstruction_sdr_db=reconstruction_sdr(reference,estimate),
                si_sdr_db=si_sdr(reference,estimate))
