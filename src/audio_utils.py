"""Validated, bounded reads of MUSDB18-HQ stereo stems."""
from pathlib import Path
import numpy as np
import soundfile as sf

REQUIRED_STEMS = ['mixture.wav', 'vocals.wav', 'drums.wav', 'bass.wav', 'other.wav']

def read_audio(path, start=0, frames=-1):
    audio, sr = sf.read(path, start=start, frames=frames, dtype='float64', always_2d=True)
    if not np.all(np.isfinite(audio)):
        raise ValueError(f'Non-finite audio: {path}')
    return audio, sr

def load_track_segment(track_dir: Path, start_sec: float, duration_sec: float):
    track_dir = Path(track_dir)
    if not np.isfinite(start_sec) or start_sec < 0:
        raise ValueError('start_sec must be finite and nonnegative')
    if not np.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError('duration_sec must be finite and positive')
    paths = [track_dir / name for name in REQUIRED_STEMS]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f'Missing stem: {p}')
    info = [sf.info(p) for p in paths]
    if len({(i.samplerate, i.channels, i.frames) for i in info}) != 1:
        raise ValueError('Stems must have identical sample rates, channels and lengths')
    sr = info[0].samplerate
    if sr != 44100 or info[0].channels != 2:
        raise ValueError('This experiment requires 44.1 kHz stereo MUSDB18-HQ stems')
    start, frames = round(start_sec * sr), round(duration_sec * sr)
    if frames <= 0 or start + frames > info[0].frames:
        raise ValueError('Requested segment extends outside the track')
    mix, v, drums, bass, other = [read_audio(p, start, frames)[0] for p in paths]
    a = drums + bass + other
    error = mix - (v + a)
    return dict(mixture=mix, vocals=v, accompaniment=a, sr=sr,
                mixture_consistency_error=float(np.max(np.abs(error))),
                mixture_relative_error=float(np.linalg.norm(error) / max(np.linalg.norm(mix), 1e-30)))

def save_comparison(audio_by_name, sr, output_dir):
    """FLOAT WAVs preserve levels; a shared gain prevents playback clipping."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    peak = max(float(np.max(np.abs(x))) for x in audio_by_name.values())
    gain = min(1.0, 0.95 / max(peak, 1e-30))
    for name, x in audio_by_name.items():
        sf.write(output_dir / f'{name}.wav', gain * x, sr, subtype='FLOAT')
    return gain
