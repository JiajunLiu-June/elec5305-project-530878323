"""Generate study tables and figures from measured CSVs, without model reruns."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from .run_experiment import dump_csv

LABELS = dict(mixture='Mixture',irm='Power IRM',magnitude_phase='True mag + mix phase',
              clipped_psm='Clipped PSM',nonnegative_psm='Nonnegative PSM',
              openunmix='UMXHQ',htdemucs='HTDemucs')
COLORS = dict(mixture='#8b96a5',irm='#357abd',magnitude_phase='#bb6a2f',
              clipped_psm='#208674',nonnegative_psm='#65a34c',
              openunmix='#8b5aa8',htdemucs='#d04959')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/study'))
    args=p.parse_args(argv)
    out=args.results
    figs=out/'figures';figs.mkdir(exist_ok=True)
    df=pd.read_csv(out/'metrics.csv')
    methods=[m for m in LABELS if m in set(df.method)]
    natural=df[df.condition=='natural']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    # Two excerpts are repeated measurements; average within each track first.
    per_track=natural.groupby(['track','method'])[['reconstruction_sdr_db','si_sdr_db']].mean().reset_index()
    dump_csv(per_track.to_dict('records'),out/'natural_track_scores.csv')
    table=[]
    for m in methods:
        g=per_track[per_track.method==m]
        table.append(dict(method=m,tracks=len(g),mean_rec_sdr_db=g.reconstruction_sdr_db.mean(),
                          min_track_rec_sdr_db=g.reconstruction_sdr_db.min(),
                          max_track_rec_sdr_db=g.reconstruction_sdr_db.max(),
                          mean_si_sdr_db=g.si_sdr_db.mean()))
    dump_csv(table,out/'natural_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,metric,title in zip(axes,['reconstruction_sdr_db','si_sdr_db'],['Reconstruction SDR','SI-SDR']):
        for i,m in enumerate(methods):
            vals=per_track.loc[per_track.method==m,metric].to_numpy()
            ax.bar(i,vals.mean(),color=COLORS[m],alpha=.78)
            ax.scatter(i+np.linspace(-.13,.13,len(vals)),vals,s=18,color='#172536',zorder=4)
        ax.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=45,ha='right')
        ax.set_ylabel('dB');ax.set_title(title);ax.axhline(0,color='#bbb',lw=.7)
    fig.suptitle('Natural mixes: bars = mean of tracks; dots = track means')
    fig.savefig(figs/'natural_scores.png',dpi=180);plt.close(fig)
    controlled=df[df.condition!='natural']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    for ax,metric,title in zip(axes,['reconstruction_sdr_db','si_sdr_db'],['Reconstruction SDR','SI-SDR']):
        for m in methods:
            g=controlled[controlled.method==m].groupby('var_db')[metric].mean()
            # Floating point gain roundoff must not split nominal VAR groups.
            g=controlled[controlled.method==m].assign(target_var=lambda z:z.var_db.round()).groupby('target_var')[metric].mean()
            ax.plot(g.index,g.values,'o-',label=LABELS[m],color=COLORS[m],ms=4)
        ax.set_xlabel('Controlled vocal-to-accompaniment ratio (dB)')
        ax.set_ylabel(title+' (dB)');ax.grid(alpha=.15)
    axes[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1))
    fig.savefig(figs/'var_sweep.png',dpi=180);plt.close(fig)
    # Acoustic measures covary under gain changes: visualize, do not claim independent causality.
    fig,axes=plt.subplots(1,3,figsize=(11.6,3.5),layout='constrained')
    correlations=[]
    for ax,variable,label in zip(axes,['var_db','overlap','phase_disagreement'],['VAR (dB)','TF overlap O','Phase disagreement D_phi']):
        for m in methods:
            if m in ('mixture','nonnegative_psm'):continue
            g=natural[natural.method==m]
            ax.scatter(g[variable],g.reconstruction_sdr_db,label=LABELS[m],color=COLORS[m],s=24,alpha=.8)
            rho=float(spearmanr(g[variable],g.reconstruction_sdr_db).statistic)
            correlations.append(dict(method=m,variable=variable,spearman_rho=rho,n_excerpts=len(g),
                                     note='Descriptive; two excerpts per track, no independent-sample inference'))
        ax.set_xlabel(label);ax.set_ylabel('Reconstruction SDR (dB)');ax.grid(alpha=.15)
    axes[-1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1))
    fig.savefig(figs/'difficulty.png',dpi=180);plt.close(fig)
    dump_csv(correlations,out/'descriptive_correlations.csv')
    sensitivity=pd.read_csv(out/'stft_sensitivity.csv')
    fig,ax=plt.subplots(figsize=(7.5,3.7),layout='constrained')
    for m in ('irm','magnitude_phase','clipped_psm','nonnegative_psm'):
        g=sensitivity[sensitivity.method==m].groupby('n_fft').reconstruction_sdr_db.mean()
        ax.plot(g.index,g.values,'o-',label=LABELS[m],color=COLORS[m])
    ax.set_xticks([1024,2048,4096]);ax.set_xlabel('Analysis FFT size (hop = FFT size / 4)')
    ax.set_ylabel('Mean reconstruction SDR (dB)');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(figs/'stft_sensitivity.png',dpi=180);plt.close(fig)
    paired=natural.pivot(index='case_id',columns='method',values='reconstruction_sdr_db')
    gain=paired.clipped_psm-paired.irm
    track_paired=per_track.pivot(index='track',columns='method',values='reconstruction_sdr_db')
    track_gain=track_paired.clipped_psm-track_paired.irm
    summary=dict(n_tracks=int(natural.track.nunique()),n_excerpts=int(natural.case_id.nunique()),
                 n_cases=int(df.case_id.nunique()),n_rows=len(df),
                 clipped_psm_gain_mean_db=float(track_gain.mean()),clipped_psm_gain_min_db=float(gain.min()),
                 clipped_psm_gain_max_db=float(gain.max()),
                 max_stft_relative_error=float(df.stft_relative_error.max()),
                 max_phase_identity_error=float(df.phase_identity_error.max()),
                 max_original_mixture_relative_error=float(df.original_mixture_relative_error.max()),
                 natural_table=table)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
