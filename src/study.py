"""Short command-line entry point for the paired separation study.

The presentation uses ``python -m src.study --help``. This delegates to the
same argument parser and implementation as ``python -m src.run_experiment``.
"""
from .run_experiment import main

if __name__ == '__main__':
    main()
