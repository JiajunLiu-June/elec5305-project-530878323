"""Targeted iteration: add six seconds of context before/after natural excerpts.

Both models process exactly the same 18-second signal; only the central six
seconds are scored. The main 36-case study remains a separate experiment.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .audio_utils import load_track_segment,save_comparison
from .models import PretrainedSeparators
from .metrics import score
from .config import DATASET_ROOT
from .run_experiment import dump_csv

def main():
    out=Path('results/study')
    d=pd.read_csv(out/'metrics.csv')
    natural=d[(d.condition=='natural')&d.method.isin(['openunmix','htdemucs'])]
    models=PretrainedSeparators()
    rows=[]
    for case,g in natural.groupby('case_id'):
        row=g.iloc[0]
        data=load_track_segment(DATASET_ROOT/'test'/row.track,row.start_sec-6,18)
        v,a=data['vocals'],data['accompaniment'];x=v+a
        center=slice(6*44100,12*44100)
        for name in ['openunmix','htdemucs']:
            y,elapsed=models.separate(name,x)
            y=y[center]
            short=g[g.method==name].iloc[0]
            s=score(v[center],y)
            rows.append(dict(case_id=case,track=row.track,start_sec=row.start_sec,
                             method=name,input_duration_sec=18,scored_duration_sec=6,
                             context_before_sec=6,context_after_sec=6,
                             short_reconstruction_sdr_db=short.reconstruction_sdr_db,
                             short_si_sdr_db=short.si_sdr_db,**s,
                             gain_rec_sdr_db=s['reconstruction_sdr_db']-short.reconstruction_sdr_db,
                             gain_si_sdr_db=s['si_sdr_db']-short.si_sdr_db,
                             inference_seconds=elapsed))
            if case=='t00_s60_d6_natural':
                save_comparison(dict(reference=v[center],context_estimate=y),44100,out/'context_audio'/name)
            print(case,name,rows[-1]['gain_rec_sdr_db'],flush=True)
            dump_csv(rows,out/'context_experiment.csv')

if __name__=='__main__':main()
