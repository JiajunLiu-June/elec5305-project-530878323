"""Validate the completed study, audio exports, manifest and PDF consistency."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from pypdf import PdfReader
from src.run_experiment import signature

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/study'
d=pd.read_csv(R/'metrics.csv');c=pd.read_csv(R/'cases.csv')
manifest=json.loads((R/'run_manifest.json').read_text())
assert manifest['code_hash']==signature()
assert len(manifest['tracks'])==5
assert d.groupby('case_id').method.nunique().eq(7).all()
assert len(d)==7*len(c)==manifest['rows']
assert len(c)==manifest['cases']
assert np.isfinite(d[['reconstruction_sdr_db','si_sdr_db']]).all().all()
assert d.stft_relative_error.max()<1e-12
assert d.phase_identity_error.max()<1e-12
control=d[d.method=='mixture']
assert np.max(np.abs(control.reconstruction_sdr_db-control.var_db))<1e-9
assert d.groupby(['track','start_sec']).condition.nunique().eq(4).all()
assert d.overlap.between(0,1).all() and d.phase_disagreement.between(0,2).all()
for cid,g in d.groupby('case_id'):
    q=g.set_index('method').coefficient_error_ratio
    assert q['nonnegative_psm']<=q['clipped_psm']+1e-10
    assert q['clipped_psm']<=q['irm']+1e-10
audio=list((R/'audio').rglob('*.wav'))
for path in audio:
    x,sr=sf.read(path,always_2d=True)
    assert sr==44100 and x.shape==(264600,2)
    assert np.isfinite(x).all() and np.max(abs(x))<=0.950001
b=pd.read_csv(R/'museval.csv')
assert len(b)==6*d[d.condition=='natural'].case_id.nunique()
assert b.finite_sdr_windows.ge(1).all()
cx=pd.read_csv(R/'context_experiment.csv')
assert len(cx)==18 and cx.case_id.nunique()==9
assert cx.groupby('case_id').method.nunique().eq(2).all()
assert np.isfinite(cx[['reconstruction_sdr_db','si_sdr_db']]).all().all()
assert np.max(abs(cx.reconstruction_sdr_db-cx.short_reconstruction_sdr_db-cx.gain_rec_sdr_db))<1e-10
pdf=ROOT/'output/pdf/ELEC5305_Project_Report_Jiajun_Liu.pdf'
reader=PdfReader(pdf)
assert len(reader.pages)==10
text='\n'.join(p.extract_text() for p in reader.pages)
assert str(len(c))+' paired' in text
assert '800 extra songs' in text
assert 'MUSDB-only' not in text
weights=[]
for f in sorted((ROOT/'.cache/torch/hub/checkpoints').iterdir()):
    digest=hashlib.sha256(f.read_bytes()).hexdigest()
    expected=f.stem.split('-')[-1]
    assert digest.startswith(expected),(f.name,digest)
    weights.append(dict(filename=f.name,sha256=digest,bytes=f.stat().st_size))
(R/'model_weight_checksums.json').write_text(json.dumps(weights,indent=2))
result=dict(status='PASS',tracks=5,active_base_excerpts=int(d[d.condition=='natural'].case_id.nunique()),
            excluded=manifest['skipped'],paired_conditions=len(c),metric_rows=len(d),
            bss_eval_rows=len(b),valid_audio_files=len(audio),pdf_pages=len(reader.pages),
            context_comparison_rows=len(cx),
            maximum_stft_relative_error=float(d.stft_relative_error.max()),
            maximum_phase_identity_error=float(d.phase_identity_error.max()),
            weights_sha256_verified=len(weights),code_hash=signature())
(R/'validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
