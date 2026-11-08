"""Summarize the predeclared all-natural-excerpt context iteration."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .audio_utils import save_comparison

def main():
    out=Path('results/study');d=pd.read_csv(out/'context_experiment.csv')
    assert len(d)==18 and d.groupby('case_id').method.nunique().eq(2).all()
    summaries=[]
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for ax,name in zip(axes,['openunmix','htdemucs']):
        g=d[d.method==name]
        track=g.groupby('track')[['gain_rec_sdr_db','gain_si_sdr_db']].mean()
        summaries.append(dict(method=name,track_mean_rec_gain_db=float(track.gain_rec_sdr_db.mean()),
                              track_mean_si_gain_db=float(track.gain_si_sdr_db.mean()),
                              excerpt_gain_min_db=float(g.gain_rec_sdr_db.min()),
                              excerpt_gain_max_db=float(g.gain_rec_sdr_db.max()),
                              improved_excerpts=int((g.gain_rec_sdr_db>0).sum()),n_excerpts=len(g)))
        ax.scatter(g.short_reconstruction_sdr_db,g.reconstruction_sdr_db,color='#357abd' if name=='openunmix' else '#d04959')
        lim=[min(0,float(g.short_reconstruction_sdr_db.min()))-1,float(g.reconstruction_sdr_db.max())+1]
        ax.plot(lim,lim,'k--',lw=1)
        ax.set(xlabel='6 s input: reconstruction SDR (dB)',ylabel='18 s input, centre scored (dB)',
               title='UMXHQ' if name=='openunmix' else 'HTDemucs',xlim=lim,ylim=lim)
        ax.grid(alpha=.15)
    fig.savefig(out/'figures/context_iteration.png',dpi=180);plt.close(fig)
    worst=d[(d.case_id=='t00_s60_d6_natural')&(d.method=='htdemucs')].iloc[0].to_dict()
    result=dict(input_seconds=18,score_seconds=6,context_before_seconds=6,context_after_seconds=6,
                same_natural_cases=9,rows=18,summary=summaries,failed_case=worst,
                limitation='Context and default Demucs split-window alignment change together; not a pure architecture or causal phase test')
    (out/'context_summary.json').write_text(json.dumps(result,indent=2))
    cid='t00_s60_d6_natural'
    with np.load(out/'cache'/f'{cid}_oracles.npz') as ck:v,a=ck['vocals'],ck['accompaniment']
    with np.load(out/'cache'/f'{cid}_htdemucs.npz') as ck:old=ck['audio']
    after,_=sf.read(out/'context_audio/htdemucs/context_estimate.wav',always_2d=True)
    saved_ref,_=sf.read(out/'context_audio/htdemucs/reference.wav',always_2d=True)
    saved_gain=np.linalg.norm(saved_ref)/np.linalg.norm(v)
    save_comparison(dict(reference=v,mixture=v+a,short_input=old,with_context=after/saved_gain),
                    44100,out/'context_audio/comparison')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
