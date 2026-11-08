"""BSS Eval v4 cross-check on the natural excerpts (two source images)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import importlib.util
import importlib.metadata
# Load the installed, unmodified numerical module directly. Importing museval's
# dataset wrappers unnecessarily requires ffprobe, even for in-memory WAV arrays.
_path=importlib.metadata.distribution('museval').locate_file('museval/metrics.py')
_spec=importlib.util.spec_from_file_location('museval_numerical_metrics',_path)
_metrics=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
bss_eval=_metrics.bss_eval
from .run_experiment import dump_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/study'))
    args=p.parse_args()
    out=args.results
    df=pd.read_csv(out/'metrics.csv')
    rows=[]
    checkpoint=out/'museval.csv'
    if checkpoint.exists():rows=pd.read_csv(checkpoint).to_dict('records')
    done={(r['case_id'],r['method']) for r in rows}
    with threadpool_limits(limits=4):
        for cid,g in df[df.condition=='natural'].groupby('case_id'):
            with np.load(out/'cache'/f'{cid}_oracles.npz') as ck:
                v,a=ck['vocals'],ck['accompaniment'];x=v+a
                ref=np.stack([v,a])
                for name in g.method:
                    # The no-separation control has an exactly zero accompaniment
                    # estimate, which museval rejects. Do not inject fake noise.
                    if name == 'mixture':continue
                    if (cid,name) in done:continue
                    if name in ck:y=ck[name]
                    else:
                        with np.load(out/'cache'/f'{cid}_{name}.npz') as m:y=m['audio']
                    sdr,isr,sir,sar,_=bss_eval(ref,np.stack([y,x-y]),window=44100,hop=44100,
                                               filters_len=512,framewise_filters=False,
                                               compute_permutation=False,bsseval_sources_version=False)
                    values=dict(case_id=cid,track=g.iloc[0].track,method=name,
                                bss_eval_v4_vocal_sdr_db=float(np.nanmedian(sdr[0])),
                                bss_eval_v4_vocal_sir_db=float(np.nanmedian(sir[0])),
                                bss_eval_v4_vocal_sar_db=float(np.nanmedian(sar[0])),
                                finite_sdr_windows=int(np.isfinite(sdr[0]).sum()),
                                window_samples=44100,hop_samples=44100,filter_length=512)
                    rows.append(values);dump_csv(rows,checkpoint)
                    print(cid,name,round(values['bss_eval_v4_vocal_sdr_db'],2),flush=True)


if __name__=='__main__':main()
