"""Paired oracle/learned vocal-separation study; all outputs are reproducible.

python -m src.run_experiment --oracle-only
python -m src.run_experiment --models openunmix htdemucs
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from .audio_utils import load_track_segment, save_comparison
from .config import DATASET_ROOT, RESULTS_ROOT
from .metrics import (vocal_to_accompaniment_ratio, tf_overlap,
                      phase_disagreement, score)
from .oracle_methods import oracle_estimates
from .stft_utils import multichannel_stft, multichannel_istft


def dump_csv(rows, path):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def signature():
    root = Path(__file__).parent
    names = ['audio_utils.py','stft_utils.py','oracle_methods.py','metrics.py','models.py','config.py']
    return hashlib.sha256(b''.join((root/n).read_bytes() for n in names)).hexdigest()


def analyse(v, a, n_fft=2048):
    x = v+a
    X,V,A = [multichannel_stft(z,n_fft=n_fft) for z in (x,v,a)]
    spectra,mask = oracle_estimates(X,V,A)
    estimates = {k:multichannel_istft(s,len(v),n_fft=n_fft) for k,s in spectra.items()}
    energy = np.sum(np.abs(V)**2)
    coefficient_error = lambda s: float(np.sum(np.abs(s-V)**2)/energy)
    dphi = phase_disagreement(V,X)
    info = dict(var_db=vocal_to_accompaniment_ratio(v,a), overlap=tf_overlap(V,A),
                phase_disagreement=dphi,
                stft_relative_error=float(np.linalg.norm(multichannel_istft(V,len(v),n_fft=n_fft)-v)/np.linalg.norm(v)),
                phase_identity_error=abs(coefficient_error(spectra['magnitude_phase'])-2*dphi),
                negative_projection_energy_fraction=float(np.sum(np.abs(V)**2*(np.real(V*np.conj(X))<0))/energy),
                cancellation_energy_fraction=float(np.sum(np.abs(V)**2*(np.abs(V)>np.abs(X)))/energy))
    errors = {k:coefficient_error(s) for k,s in spectra.items()}
    if errors['clipped_psm'] > errors['irm'] + 1e-10:
        raise AssertionError('Phase-sensitive mask violated its coefficient optimum')
    return estimates,info,errors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=DATASET_ROOT)
    p.add_argument('--output',type=Path,default=RESULTS_ROOT/'study')
    p.add_argument('--track',default=None)
    p.add_argument('--limit-tracks',type=int,default=5)
    p.add_argument('--starts',type=float,nargs='+',default=[30.,60.])
    p.add_argument('--duration',type=float,default=6.)
    p.add_argument('--var',type=float,nargs='+',default=[-10.,0.,10.])
    p.add_argument('--oracle-only',action='store_true')
    p.add_argument('--models',nargs='*',default=['openunmix','htdemucs'])
    args = p.parse_args(argv)
    test_root = args.dataset_root/'test'
    if not test_root.is_dir():
        raise FileNotFoundError(f'{test_root} not found; run tools/download_subset.py first')
    tracks = sorted(t for t in test_root.iterdir() if t.is_dir())[:args.limit_tracks]
    if args.track:
        tracks = [test_root/args.track]
    if not tracks:
        raise RuntimeError('No test tracks found')
    out = args.output
    cache = out/'cache'
    cache.mkdir(parents=True,exist_ok=True)
    wanted = [] if args.oracle_only else args.models
    models = None
    rows, sensitivity, cases, skipped = [],[],[],[]
    code_hash = signature()
    versions = {k:importlib.metadata.version(k) for k in
                ['numpy','scipy','librosa','soundfile','torch','torchaudio','openunmix','demucs','museval']}
    for ti,track in enumerate(tracks):
        for start in args.starts:
            data = load_track_segment(track,start,args.duration)
            v, original_a = data['vocals'],data['accompaniment']
            rms = float(np.sqrt(np.mean(v*v)))
            if rms < 10**(-50/20):
                skipped.append(dict(track=track.name,start=start,reason='vocal RMS below -50 dBFS',rms=rms))
                continue
            native_var = vocal_to_accompaniment_ratio(v,original_a)
            for condition,target in [('natural',None)]+[(f'var{z:+g}',z) for z in args.var]:
                gain = 1. if target is None else 10**((native_var-target)/20)
                a = original_a*gain
                x = v+a
                case_id = f't{ti:02d}_s{start:g}_d{args.duration:g}_{condition}'
                print(f'{case_id}: {track.name}',flush=True)
                estimates,info,errors = analyse(v,a)
                base = dict(case_id=case_id,track=track.name,start_sec=start,duration_sec=args.duration,
                            condition=condition,accompaniment_gain=gain,native_var_db=native_var,
                            original_mixture_relative_error=data['mixture_relative_error'],**info)
                cases.append(base)
                estimates = dict(mixture=x,**estimates)
                for name in wanted:
                    checkpoint = cache/f'{case_id}_{name}.npz'
                    input_hash = hashlib.sha256(x.tobytes()).hexdigest()
                    if checkpoint.exists():
                        with np.load(checkpoint,allow_pickle=False) as ck:
                            if str(ck['code_hash']) != code_hash or str(ck['input_hash']) != input_hash:
                                raise RuntimeError(f'Stale checkpoint: {checkpoint}; use a new output directory')
                            estimates[name],elapsed = ck['audio'],float(ck['seconds'])
                    else:
                        if models is None:
                            from .models import PretrainedSeparators
                            models = PretrainedSeparators(wanted)
                        estimates[name],elapsed = models.separate(name,x)
                        np.savez_compressed(checkpoint,audio=estimates[name],seconds=elapsed,
                                            code_hash=code_hash,input_hash=input_hash)
                    print(f'  {name}: {elapsed:.2f} s',flush=True)
                    base[f'{name}_seconds'] = elapsed
                for name,y in estimates.items():
                    rows.append(dict(**base,method=name,**score(v,y),
                                     coefficient_error_ratio=errors.get(name,''),
                                     inference_seconds=base.get(f'{name}_seconds','')))
                # Save every oracle waveform as well, for subsequent independent evaluation.
                np.savez_compressed(cache/f'{case_id}_oracles.npz',vocals=v,accompaniment=a,
                                    **{k:y for k,y in estimates.items() if k not in wanted})
                # Fixed first excerpt at natural balance: no output-based demonstration selection.
                if start == args.starts[0] and condition == 'natural':
                    playback_gain = save_comparison(dict(reference=v,accompaniment=a,**estimates),
                                                    data['sr'],out/'audio'/f't{ti:02d}')
                    base['playback_gain'] = playback_gain
                dump_csv(rows,out/'metrics.csv')
                dump_csv(cases,out/'cases.csv')
            for n_fft in (1024,2048,4096):
                est,info,errors = analyse(v,original_a,n_fft=n_fft)
                for name,y in est.items():
                    sensitivity.append(dict(track=track.name,start_sec=start,n_fft=n_fft,
                                            **info,method=name,**score(v,y)))
    dump_csv(sensitivity,out/'stft_sensitivity.csv')
    manifest = dict(date='2026-11-08',tracks=[t.name for t in tracks],starts=args.starts,
                    duration=args.duration,var_targets=args.var,conditions='natural plus controlled VAR',
                    mixture='Always vocals + gain*(drums+bass+other); identical for all methods',
                    analysis_stft=dict(fs=44100,n_fft=2048,hop=512,window='hann',center=True),
                    models=wanted,openunmix='UMXHQ, four targets, niter=1, native 4096/1024 STFT',
                    htdemucs='htdemucs trained on MUSDB + 800 extra songs, CPU, shifts=0, split=True, overlap=0.25',
                    device='CPU, 4 torch threads',python=platform.python_version(),
                    platform=platform.platform(),versions=versions,code_hash=code_hash,
                    cases=len(cases),rows=len(rows),skipped=skipped)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Completed: {len(cases)} paired cases, {len(rows)} metric rows. {out}',flush=True)


if __name__ == '__main__':
    main()
