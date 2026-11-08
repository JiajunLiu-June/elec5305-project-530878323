"""Unmodified author implementations and official pretrained weights (HTDemucs also used 800 extra songs)."""
import time
import numpy as np
import torch

class PretrainedSeparators:
    def __init__(self, names=('openunmix','htdemucs'), threads=4):
        torch.set_num_threads(threads)
        self.models = {}
        for name in names:
            if name == 'openunmix':
                from openunmix import umxhq
                model = umxhq(niter=1, device='cpu').eval()
            elif name == 'htdemucs':
                from demucs.pretrained import get_model
                model = get_model('htdemucs').cpu().eval()
            else:
                raise ValueError(f'Unknown model: {name}')
            self.models[name] = model

    def separate(self, name, mixture):
        model = self.models[name]
        x = torch.from_numpy(np.asarray(mixture.T, dtype=np.float32).copy())[None]
        started = time.perf_counter()
        with torch.inference_mode():
            if name == 'openunmix':
                y = model(x)[0, list(model.target_models).index('vocals')]
            else:
                from demucs.apply import apply_model
                # Match the official CLI's shared mono mean/std preprocessing.
                mono = x.mean(dim=1)
                mean, std = mono.mean(), mono.std()
                if std < 1e-8:
                    raise ValueError('Mixture is effectively silent')
                normalized = (x-mean)/std
                # shifts=0 removes random test-time augmentation; no extra-data model.
                estimates = apply_model(model,normalized,shifts=0,split=True,
                                        overlap=0.25,device='cpu',progress=False)
                y = estimates[0, model.sources.index('vocals')]*std + mean
        out = y.cpu().numpy().T.astype(np.float64)
        if out.shape != mixture.shape or not np.isfinite(out).all():
            raise RuntimeError(f'Invalid output from {name}')
        return out, time.perf_counter()-started
