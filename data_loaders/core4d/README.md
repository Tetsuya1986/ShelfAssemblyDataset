# CORE4D Dataset - ShelfAssembly Compatible Format

## Overview

This module provides the `CORE4DDataset` class that wraps the CORE4D dataset and makes it compatible with the ShelfAssembly dataset format. The dataset outputs motion features in a standardized format suitable for motion generation and analysis tasks.

## Key Changes

The modified `dataset_hho.py` file has been updated to:

1. **Extract motion features in ShelfAssembly format**: Converts SMPL pose parameters to 6D rotation representations
2. **Standardize output structure**: Returns motion and annotation dictionaries similar to ShelfAssembly
3. **Support clips extraction**: Automatically extracts motion clips with specified past/future frame lengths
4. **Handle multi-person data**: Processes both Person1 and Person2 motions separately

## Output Format

### Motion Dictionary
Each motion clip returns the following features:

| Feature | Shape | Description |
|---------|-------|-------------|
| `global_orient` | `(N_frames, 6)` | Global orientation in 6D rotation format |
| `body_pose` | `(N_frames, 252)` | Body pose for 42 joints in 6D rotation format |
| `left_hand_pose` | `(N_frames, 60)` | Left hand pose for 10 joints in 6D rotation format |
| `right_hand_pose` | `(N_frames, 60)` | Right hand pose for 10 joints in 6D rotation format |
| `root_pos` | `(N_frames, 3)` | Root translation (3D position) |
| `clip_name` | str | Clip name identifier |
| `seq_name` | str | Sequence name identifier |
| `person_id` | str | Person identifier (Person1 or Person2) |
| `seq_idx` | int | Sequence index |
| `no` | int | Sequence number |

### Annotation Dictionary
| Field | Type | Description |
|-------|------|-------------|
| `no` | int | Sequence number |
| `clip_name` | str | Clip name |
| `seq_name` | str | Sequence name |
| `person_id` | str | Person identifier |
| `valid_length` | int | Number of valid frames in the clip |
| `dataset` | str | Dataset name ('CORE4D') |

## Usage Example

```python
from data_loaders.core4d.dataset_hho import CORE4DDataset

# Initialize dataset
dataset = CORE4DDataset(
    mode='train',
    past_len=15,           # Number of past frames
    future_len=15,         # Number of future frames
    sample_rate=1,         # Frame sampling rate
    dataset_root="/path/to/core4d/dataset",
    test_set="all"
)

# Get a single sample
motion, annotation = dataset[0]

# Access motion features
print(motion['global_orient'].shape)  # (30, 6)
print(motion['body_pose'].shape)      # (30, 252)
print(motion['root_pos'].shape)       # (30, 3)

# Get annotation info
print(f"Sequence: {annotation['seq_name']}")
print(f"Person: {annotation['person_id']}")
print(f"Valid frames: {annotation['valid_length']}")
```

## Rotation Representation

The dataset uses **6D rotation representation** instead of the raw axis-angle representation. This is achieved through:

1. Converting axis-angle to rotation matrices
2. Extracting the first two columns of the rotation matrix (6 values)

This representation is numerically stable and commonly used in motion generation tasks. See `utils.rotation_conversions` for the conversion functions.

## Integration with ShelfAssembly Pipeline

The CORE4D dataset can now be used interchangeably with ShelfAssembly in your pipeline:

```python
from data_loaders.shelf_assembly.dataset import ShelfAssemblyDataset
from data_loaders.core4d.dataset_hho import CORE4DDataset

# Both datasets have the same output format
shelf_dataset = ShelfAssemblyDataset(mode='train')
core4d_dataset = CORE4DDataset(mode='train', dataset_root="...")

# Same data format allows mixing or sequential use
motion_shelf, annot_shelf = shelf_dataset[0]
motion_core4d, annot_core4d = core4d_dataset[0]

# Both have compatible structure
assert motion_shelf['global_orient'].shape == motion_core4d['global_orient'].shape
```

## Parameters

### Constructor Arguments

- `mode` (str): 'train' or 'test' - Dataset split
- `past_len` (int): Number of past frames to include (default: 15)
- `future_len` (int): Number of future frames to include (default: 15)
- `sample_rate` (int): Frame sampling rate (default: 1, no sampling)
- `dataset_root` (str): Path to CORE4D dataset root directory
- `smplx_model_dir` (str): Path to SMPLX model files
- `test_set` (str): For test mode: 'all', 'seen', or 'unseen' (default: 'all')
- `**kwargs`: Additional arguments for future extensions

## Data Organization

The dataset expects the following structure:

```
dataset_root/
├── {clip_name}/
│   ├── {seq_name}/
│   │   ├── data.npz          # CORE4D data file
│   │   ├── human_normal.npz  # (optional) precomputed normals
│   │   └── ...
│   └── ...
└── ...
```

## Notes

1. **6D Rotation Format**: The 6D representation uses the first two columns of the 3×3 rotation matrix. To convert back to other representations, use:
   - `rotation_6d_to_matrix()` (if available)
   - Manual matrix reconstruction via Gram-Schmidt

2. **Clipping Strategy**: Motion clips are extracted with:
   - `fragment = (past_len + future_len) * sample_rate`
   - For each sequence, clips are generated at regular intervals

3. **Object Categories**: Only sequences with these object categories are included:
   - chair, desk, board, box, bucket, stick

4. **Error Handling**: The dataset gracefully handles missing or corrupted files and prints warnings

## Related Files

- `dataset.py` (ShelfAssembly): Reference implementation for output format
- `utils/rotation_conversions.py`: Rotation conversion utilities
- `data_loaders/humanml/utils/get_opt.py`: Configuration utilities

## Compatibility

- **PyTorch**: Uses `torch.utils.data.Dataset` base class
- **NumPy**: Data processing and manipulation
- **SciPy**: Rotation handling
- **PyYAML**: Configuration files

## Future Extensions

Possible enhancements:
- Add augmentation support (random rotation, scaling, etc.)
- Support for conditional generation with text/action labels
- Multi-modal input support (video frames, point clouds)
- Batched preprocessing for faster loading
