# Local Path Configuration for UnSAMFlow

This document describes the changes made to convert UnSAMFlow from using Meta's internal manifold file system to standard Linux local paths.

## Changes Made

### 1. Created Local Path Configuration (`utils/local_paths.py`)

- **Base Directory**: `/workspace/UnSAMFlow_data`
- **Model Directory**: `/workspace/UnSAMFlow/models`
- **Checkpoint Directory**: `/workspace/UnSAMFlow/checkpoints`
- **Results Directory**: `/workspace/UnSAMFlow/results`

### 2. Updated Files

#### Core Files:
- `test.py` - Removed manifold imports, updated path handling
- `train.py` - Removed manifold dependencies, simplified file operations
- `utils/torch_utils.py` - Updated checkpoint loading/saving to use local paths
- `utils/config_parser.py` - Added local path conversion
- `datasets/flow_datasets.py` - Updated to use local paths
- `trainer/base_trainer.py` - Removed manifold imports
- `trainer/kitti_trainer_ar.py` - Updated path handling
- `sam_inference.py` - Replaced hardcoded paths with configuration

#### Configuration Files:
- `configs/kitti_base.json` - Updated dataset paths to use local directories
- `configs/sintel_base.json` - Already had correct local paths

### 3. Directory Structure

The following directory structure is expected:

```
/workspace/
├── UnSAMFlow_data/
│   ├── KITTI-2012/
│   ├── KITTI-2015/
│   ├── KITTI-raw/
│   ├── Sintel/
│   ├── Sintel-raw/
│   ├── KITTI-2012_seg/
│   ├── KITTI-2015_seg/
│   └── Sintel_seg/
├── UnSAMFlow/
│   ├── models/
│   ├── checkpoints/
│   └── results/
```

## Setup Instructions

1. **Run the setup script** to create the directory structure:
   ```bash
   cd UnSAMFlow
   python setup_local_paths.py
   ```

2. **Place your datasets** in the appropriate subdirectories under `/workspace/UnSAMFlow_data/`

3. **Update paths if needed** in `utils/local_paths.py` if you want to use different base directories

## Key Functions

### `get_local_path(manifold_path)`
Converts manifold-style paths to local paths:
- `manifold://bucket/path` → `/workspace/UnSAMFlow_data/path`
- `memcache_manifold://bucket/path` → `/workspace/UnSAMFlow_data/path`
- Regular paths are returned unchanged

### `ensure_dir(path)`
Creates directories if they don't exist.

## Usage

The code now works with standard Linux file operations:
- File loading uses `torch.load()` directly
- File saving uses `torch.save()` directly
- Directory creation uses `os.makedirs()`
- No special manifold file system dependencies

## Migration Notes

- All manifold-specific imports have been removed
- `pathmgr` operations replaced with standard `os` operations
- `MANIFOLD_BUCKET` and `MANIFOLD_PATH` constants removed
- Hardcoded `YOUR_DIR` references replaced with configuration-based paths

## Troubleshooting

If you encounter path-related issues:

1. Check that the directories exist: `python setup_local_paths.py`
2. Verify dataset paths in configuration files
3. Ensure proper permissions on the data directories
4. Update `utils/local_paths.py` if you need different base paths 