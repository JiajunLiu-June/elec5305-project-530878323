"""A short real execution for the project demonstration, with timestamped logs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .audio_utils import load_track_segment,save_comparison
from .config import DATASET_ROOT
from .run_experiment import analyse
from .models import PretrainedSeparators
from .metrics import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/demo'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    events=[];began=time.perf_counter()
    def log(message):
        events.append(dict(seconds=time.perf_counter()-began,message=message))
        print(message,flush=True)
    log('$ python -m src.demo')
    track=sorted((DATASET_ROOT/'test').iterdir())[0]
    data=load_track_segment(track,30,6)
    v,a=data['vocals'],data['accompaniment'];x=v+a
    log(f'Track: {track.name}')
    log('Read 30-36 seconds: 264600 samples x 2 channels, 44100 Hz')
    est,info,_=analyse(v,a)
    log(f'STFT relative reconstruction error: {info["stft_relative_error"]:.2e}')
    log(f'VAR {info["var_db"]:.2f} dB | O {info["overlap"]:.4f} | D_phi {info["phase_disagreement"]:.4f}')
    log('Loading official pretrained UMXHQ and HTDemucs...')
    models=PretrainedSeparators()
    for name in ['openunmix','htdemucs']:
        est[name],seconds=models.separate(name,x)
        log(f'{name}: actual inference completed in {seconds:.2f} seconds')
    log('Method              Rec SDR     SI-SDR (dB)')
    for name,y in est.items():
        s=score(v,y)
        log(f'{name:19s} {s["reconstruction_sdr_db"]:7.2f}     {s["si_sdr_db"]:7.2f}')
    gain=save_comparison(dict(reference=v,mixture=x,**est),44100,args.output/'audio')
    log(f'Saved FLOAT WAV comparisons with common gain {gain:.4f}')
    log('Complete. Oracles use true stems; pretrained models receive mixture only.')
    (args.output/'execution_events.json').write_text(json.dumps(events,indent=2))


if __name__=='__main__':main()
