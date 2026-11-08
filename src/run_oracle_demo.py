"""Single-track oracle entry point. Use python -m src.run_oracle_demo."""
import sys
from .run_experiment import main
if __name__ == "__main__":
    main(["--oracle-only","--limit-tracks","1", *sys.argv[1:]])
