from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "MUSDB18-HQ"

RESULTS_ROOT = PROJECT_ROOT / "results"


# ============================================================
# STFT configuration
# ============================================================

SAMPLE_RATE = 44100

N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 512
WINDOW = "hann"

EPS = 1e-10


# ============================================================
# Default analysis segment
# ============================================================

DEFAULT_START_SEC = 30.0
DEFAULT_DURATION_SEC = 10.0