"""Package the current deliverables, without environments, weights or raw tracks."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT.parent/'ELEC5305_Project_Jiajun_Liu_20261108.zip'
PREFIX='ELEC5305_Project_Jiajun_Liu'
files=[]
for dirname in ['src','tests','tools','docs','.github']:
    files.extend(p for p in (ROOT/dirname).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files.extend(ROOT/name for name in ['README.md','.gitignore','requirements.txt','requirements-report.txt','requirements-lock.txt'])
files.extend(p for p in (ROOT/'output/pdf').iterdir() if p.suffix in ('.pdf','.md'))
files.extend(p for p in (ROOT/'output/video').iterdir() if p.suffix in ('.mp4','.json','.md'))
files.extend(p for p in (ROOT/'results/study').rglob('*')
             if p.is_file() and 'cache' not in p.parts and p.suffix in ('.csv','.json','.txt','.png','.wav'))
files.extend(p for p in (ROOT/'results/demo').rglob('*') if p.is_file() and p.suffix in ('.json','.wav'))
with zipfile.ZipFile(TARGET,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(set(files)):
        z.write(p, f'{PREFIX}/{p.relative_to(ROOT)}')
with zipfile.ZipFile(TARGET) as z:
    assert z.testzip() is None
    assert not any('/.venv/' in n or '/data/' in n or '/cache/' in n for n in z.namelist())
result=dict(file=str(TARGET),bytes=TARGET.stat().st_size,members=len(set(files)),
            sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),zip_crc_check='PASS',
            note='Download official test data and pretrained weights with the included tools to rerun experiments')
print(json.dumps(result,indent=2))
(ROOT.parent/'ELEC5305_Project_Jiajun_Liu_20261108_manifest.json').write_text(json.dumps(result,indent=2))
