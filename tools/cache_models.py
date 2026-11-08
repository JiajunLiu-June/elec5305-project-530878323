"""Download the two declared pretrained references into the project cache."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch')
import torch
from openunmix import umxhq
from demucs.pretrained import get_model

torch.set_num_threads(4)
print('Loading official UMXHQ (four targets, one Wiener iteration)', flush=True)
m = umxhq(niter=1, device='cpu').eval()
print('UMXHQ ready', flush=True)
del m
print('Loading official HTDemucs (official checkpoint; MUSDB + 800 extra training songs)', flush=True)
m = get_model('htdemucs')
print('HTDemucs ready', m.sources, flush=True)
