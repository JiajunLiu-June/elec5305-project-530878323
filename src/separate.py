"""Separate an arbitrary WAV without ground-truth stems.

python -m src.separate song.wav --model htdemucs --output results/my_song
"""
import argparse
import json
from pathlib import Path
import numpy as np
import librosa
from .audio_utils import read_audio, save_comparison
from .models import PretrainedSeparators


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--model',choices=['openunmix','htdemucs'],default='htdemucs')
    p.add_argument('--output',type=Path,default=Path('results/separated'))
    args=p.parse_args()
    x,sr=read_audio(args.input)
    if x.shape[1]==1:x=np.repeat(x,2,axis=1)
    if x.shape[1]!=2:raise ValueError('Use a mono or stereo input')
    if sr!=44100:x=librosa.resample(x.T,orig_sr=sr,target_sr=44100).T
    models=PretrainedSeparators([args.model])
    v,elapsed=models.separate(args.model,x)
    gain=save_comparison(dict(mixture=x,vocals=v,accompaniment=x-v),44100,args.output)
    (args.output/'metadata.json').write_text(json.dumps(dict(model=args.model,
        input=str(args.input),input_sample_rate=sr,output_sample_rate=44100,
        shared_playback_gain=gain,inference_seconds=elapsed,
        note='Accompaniment is mixture minus estimated vocals; no true stems used'),indent=2))
    print(f'Saved vocals and accompaniment to {args.output}; model time {elapsed:.2f} s')


if __name__=='__main__':main()
