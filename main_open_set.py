"""
Main script for running Open-Set Recognition (OpenMax)
This script simply calls the OpenMax training/evaluation function
defined in train_openmax.py.
"""

from train_openmax import main as run_openmax


if __name__ == "__main__":
    print("\n===== Running Open-Set Recognition (OpenMax Stage) =====\n")
    run_openmax()
