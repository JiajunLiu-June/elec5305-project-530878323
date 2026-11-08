"""Measured attenuation/alignment diagnostics; no subjective listening labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .run_experiment import dump_csv

def main():
    out=Path('results/study');d=pd.read_csv(out/'metrics.csv');rows=[]
    for row in d[d.condition=='natural'].itertuples():
        with np.load(out/'cache'/f'{row.case_id}_oracles.npz') as ck:
            v=ck['vocals']
            y=ck[row.method] if row.method in ck else None
        if y is None:
            with np.load(out/'cache'/f'{row.case_id}_{row.method}.npz') as ck:y=ck['audio']
        vd=v-v.mean(axis=0);yd=y-y.mean(axis=0)
        ratio=float(np.linalg.norm(yd)/np.linalg.norm(vd))
        cosine=float(np.sum(vd*yd)/max(np.linalg.norm(vd)*np.linalg.norm(yd),1e-30))
        rows.append(dict(case_id=row.case_id,track=row.track,start_sec=row.start_sec,
                         method=row.method,reference_rms=float(np.sqrt(np.mean(v*v))),
                         estimate_rms=float(np.sqrt(np.mean(y*y))),
                         demeaned_rms_ratio=ratio,cosine_similarity=cosine,
                         reconstruction_sdr_db=row.reconstruction_sdr_db,si_sdr_db=row.si_sdr_db))
    dump_csv(rows,out/'failure_diagnostics.csv')
    learned=[r for r in rows if r['method'] in ('openunmix','htdemucs')]
    worst=min(learned,key=lambda r:r['si_sdr_db'])
    (out/'worst_learned_case.json').write_text(json.dumps(worst,indent=2))
    print(json.dumps(worst,indent=2))

if __name__=='__main__':main()
