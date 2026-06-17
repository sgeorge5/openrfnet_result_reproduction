"""
Main script for running closed-set supervised contrastive training
This script simply calls the training function defined in train_supcon.py.
"""

from train_supcon import main as train_closed_set


if __name__ == "__main__":
    print("\n===== Running Closed-Set Training (SupCon + CE) =====\n")
    train_closed_set()
