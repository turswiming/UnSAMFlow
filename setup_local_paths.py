#!/usr/bin/env python3
"""
Setup script to create local directory structure for UnSAMFlow
"""

import os
from utils.local_paths import BASE_DIR, MODEL_BASE_DIR, CHECKPOINT_BASE_DIR, RESULTS_BASE_DIR

def create_directory_structure():
    """Create the necessary directory structure for local paths"""
    
    directories = [
        BASE_DIR,
        MODEL_BASE_DIR,
        CHECKPOINT_BASE_DIR,
        RESULTS_BASE_DIR,
        os.path.join(BASE_DIR, "KITTI-2012"),
        os.path.join(BASE_DIR, "KITTI-2015"),
        os.path.join(BASE_DIR, "KITTI-raw"),
        os.path.join(BASE_DIR, "Sintel"),
        os.path.join(BASE_DIR, "Sintel-raw"),
        os.path.join(BASE_DIR, "KITTI-2012_seg"),
        os.path.join(BASE_DIR, "KITTI-2015_seg"),
        os.path.join(BASE_DIR, "Sintel_seg"),
        "results",
    ]
    
    print("Creating directory structure for UnSAMFlow...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\nDirectory structure created successfully!")
    print(f"\nBase data directory: {BASE_DIR}")
    print(f"Model directory: {MODEL_BASE_DIR}")
    print(f"Checkpoint directory: {CHECKPOINT_BASE_DIR}")
    print(f"Results directory: {RESULTS_BASE_DIR}")
    print("\nPlease place your datasets in the appropriate subdirectories under the base data directory.")

if __name__ == "__main__":
    create_directory_structure() 