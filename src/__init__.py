"""ELEC5305 vocal-separation experiment package."""
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
for _key, _sub in [('MPLCONFIGDIR','matplotlib'), ('NUMBA_CACHE_DIR','numba'), ('TORCH_HOME','torch')]:
    os.environ.setdefault(_key, str(_ROOT / '.cache' / _sub))
