"""Make common-scale spectrograms and a coefficient-error sanity figure."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display
from .stft_utils import multichannel_stft
from .summarize import LABELS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/study'))
    args=p.parse_args();out=args.results
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    df=pd.read_csv(out/'metrics.csv')
    # Lowest learned-model SI-SDR, selected by a fixed objective rule.
    hard=df[(df.condition=='natural')&df.method.isin(['openunmix','htdemucs'])].sort_values('si_sdr_db').iloc[0]
    cid=hard.case_id
    with np.load(out/'cache'/f'{cid}_oracles.npz') as ck:
        audio={k:ck[k] for k in ['vocals','mixture','accompaniment','irm','magnitude_phase','clipped_psm']}
    for name in ['openunmix','htdemucs']:
        cp=out/'cache'/f'{cid}_{name}.npz'
        if cp.exists():
            with np.load(cp) as ck:audio[name]=ck['audio']
    specs={name:np.mean(np.abs(multichannel_stft(x)),axis=0) for name,x in audio.items()}
    common_ref=max(np.max(s) for s in specs.values())
    fig,axes=plt.subplots(4,2,figsize=(11,9),layout='constrained')
    for ax,(name,s) in zip(axes.ravel(),specs.items()):
        db=20*np.log10(np.maximum(s,1e-10)/common_ref)
        im=librosa.display.specshow(db,sr=44100,hop_length=512,x_axis='time',y_axis='log',
                                    vmin=-70,vmax=0,cmap='magma',ax=ax)
        ax.set_title({'vocals':'True vocals','accompaniment':'Accompaniment'}.get(name,LABELS.get(name,name)),fontsize=10)
        ax.set_ylim(80,16000);ax.set_xlabel('Time (s)');ax.set_ylabel('Hz')
    for ax in axes.ravel()[len(specs):]:ax.axis('off')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='dB, one reference for every panel',shrink=.6,pad=.02)
    fig.suptitle(f'Lowest learned SI-SDR: {hard.track}, {hard.start_sec:g} s',fontsize=12)
    fig.savefig(figdir/'failure_spectrograms.png',dpi=180);plt.close(fig)
    (out/'failure_case.txt').write_text(f'{cid}\n{hard.track}\nStart {hard.start_sec:g} s; duration {hard.duration_sec:g} s\nSelected by lowest natural learned-model SI-SDR. This is an illustrative failure, not an unbiased average.\n')
    # The exact identity is a coefficient-domain diagnostic; waveform error is lower after projection.
    g=df[df.method=='magnitude_phase']
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.7),layout='constrained')
    axes[0].scatter(2*g.phase_disagreement,g.coefficient_error_ratio,s=20,c='#208674')
    lim=float(max(2*g.phase_disagreement.max(),g.coefficient_error_ratio.max()))*1.05
    axes[0].plot([0,lim],[0,lim],'k--',lw=1,label='Exact coefficient identity')
    axes[0].set(xlabel='2 D_phi',ylabel='Normalized coefficient error',title='True magnitude with mixture phase')
    axes[0].legend(fontsize=8)
    wave=10**(-g.reconstruction_sdr_db/10)
    axes[1].scatter(2*g.phase_disagreement,wave,s=20,c='#357abd')
    axes[1].plot([0,lim],[0,lim],'k--',lw=1)
    axes[1].set(xlabel='2 D_phi',ylabel='Normalized waveform error',title='ISTFT projection changes the error')
    fig.savefig(figdir/'phase_identity.png',dpi=180);plt.close(fig)
    # Analysis sequence, drawn locally as a code-native research figure.
    fig,ax=plt.subplots(figsize=(10,2.4),layout='constrained');ax.axis('off')
    boxes=[(.01,.34,.19,.45,'MUSDB18-HQ test stems\n5 tracks; two excerpts'),
           (.26,.34,.19,.45,'Natural + VAR remixes\nx = v + g a'),
           (.51,.34,.2,.45,'Oracle / learned systems\nidentical inputs'),
           (.77,.34,.21,.45,'SDR + SI-SDR + BSS Eval\npaired error analysis')]
    from matplotlib.patches import FancyBboxPatch
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',facecolor='#edf3f8',edgecolor='#4f708b'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8.5)
    for a,b in zip(boxes,boxes[1:]):
        ax.annotate('',xy=(b[0]-.012,.565),xytext=(a[0]+a[2]+.012,.565),arrowprops=dict(arrowstyle='->',color='#4f708b'))
    ax.text(.5,.13,'Reference stems enter only the oracle branches, difficulty measures and evaluation.',ha='center',fontsize=9)
    fig.savefig(figdir/'pipeline.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
