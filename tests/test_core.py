"""Scientific invariants, boundary handling and adversarial cancellation."""
import numpy as np
import pytest
import soundfile as sf
from src.audio_utils import load_track_segment, REQUIRED_STEMS
from src.stft_utils import multichannel_stft, multichannel_istft
from src.oracle_methods import oracle_estimates, phase_sensitive_mask
from src.metrics import phase_disagreement, reconstruction_sdr, si_sdr, tf_overlap


@pytest.mark.parametrize('length',[4096,5001,22050])
def test_stft_perfect_reconstruction(length):
    x=np.random.default_rng(3).normal(size=(length,2))
    y=multichannel_istft(multichannel_stft(x),len(x))
    assert y.shape == x.shape
    assert np.linalg.norm(y-x)/np.linalg.norm(x)<1e-12


def test_oracle_phase_identity_and_optimum():
    rng=np.random.default_rng(10)
    v=rng.normal(size=(2,18,20))+1j*rng.normal(size=(2,18,20))
    a=rng.normal(size=v.shape)+1j*rng.normal(size=v.shape)
    est,mask=oracle_estimates(v+a,v,a)
    error=lambda z:np.sum(abs(z-v)**2)
    assert error(est['magnitude_phase'])/np.sum(abs(v)**2)==pytest.approx(2*phase_disagreement(v,v+a))
    assert error(est['clipped_psm'])<=error(est['irm'])
    assert error(est['nonnegative_psm'])<=error(est['clipped_psm'])
    assert np.all((mask>=0)&(mask<=1))


def test_destructive_interference_requires_amplification():
    v=np.array([1+0j]); x=np.array([0.25+0j])
    assert phase_sensitive_mask(x,v)[0] == 1
    assert phase_sensitive_mask(x,v,upper=np.inf)[0] == 4


def test_opposite_phase_requires_sign_change():
    assert phase_sensitive_mask(np.array([-1+0j]),np.array([1+0j]))[0] == 0


def test_silence_mask_is_finite():
    z=np.zeros((2,10,10),complex)
    est,mask=oracle_estimates(z,z,z)
    assert np.all(mask==0)
    assert all(np.isfinite(y).all() for y in est.values())
    with pytest.raises(ValueError):tf_overlap(z,z)


def test_sdr_distinguishes_attenuation_from_si_sdr():
    x=np.random.default_rng(1).normal(size=(3000,2))
    assert reconstruction_sdr(x,0.5*x)==pytest.approx(6.020599913)
    assert si_sdr(x,0.5*x)>200


@pytest.fixture
def track(tmp_path):
    x=np.random.default_rng(5).normal(scale=0.01,size=(44100,2))
    for name in REQUIRED_STEMS:sf.write(tmp_path/name,x,44100,subtype='FLOAT')
    return tmp_path


@pytest.mark.parametrize('start,duration',[(-1,0.1),(0,0),(0,-1),(0.9,0.2)])
def test_invalid_intervals(track,start,duration):
    with pytest.raises(ValueError):load_track_segment(track,start,duration)


def test_loader_reads_exact_interval_and_rejects_misalignment(track):
    d=load_track_segment(track,0.1,0.2)
    assert d['vocals'].shape==(8820,2)
    sf.write(track/'bass.wav',np.ones((40000,2)),44100,subtype='FLOAT')
    with pytest.raises(ValueError):load_track_segment(track,0.1,0.2)
