"""
Local path configuration to replace manifold paths
"""

import os

# Base directory for data and results
BASE_DIR = "/workspace/UnSAMFlow_data"

# Model and checkpoint paths
MODEL_BASE_DIR = "/workspace/UnSAMFlow/models"
CHECKPOINT_BASE_DIR = "/workspace/UnSAMFlow/checkpoints"

# Results directory
RESULTS_BASE_DIR = "/workspace/UnSAMFlow/results"

# Dataset paths
DATASET_BASE_DIR = "/workspace/UnSAMFlow_data"

def get_local_path(manifold_path):
    """
    Convert manifold path to local path
    Args:
        manifold_path: Original manifold path
    Returns:
        Local path
    """
    if manifold_path.startswith("manifold://"):
        # Remove manifold:// prefix and convert to local path
        path_parts = manifold_path.replace("manifold://", "").split("/")
        if len(path_parts) >= 2:
            bucket = path_parts[0]
            remaining_path = "/".join(path_parts[1:])
            return os.path.join(BASE_DIR, remaining_path)
        else:
            return os.path.join(BASE_DIR, manifold_path.replace("manifold://", ""))
    elif manifold_path.startswith("memcache_manifold://"):
        # Remove memcache_manifold:// prefix and convert to local path
        path_parts = manifold_path.replace("memcache_manifold://", "").split("/")
        if len(path_parts) >= 2:
            bucket = path_parts[0]
            remaining_path = "/".join(path_parts[1:])
            return os.path.join(BASE_DIR, remaining_path)
        else:
            return os.path.join(BASE_DIR, manifold_path.replace("memcache_manifold://", ""))
    else:
        return manifold_path

def ensure_dir(path):
    """
    Ensure directory exists
    """
    os.makedirs(path, exist_ok=True)
    return path 