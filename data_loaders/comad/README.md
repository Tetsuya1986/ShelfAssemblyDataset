# CoMaD Dataset - ShelfAssembly Compatible Format

## Overview

This module provides the `CoMaDDataset` class that wraps the CoMaD (Coordinated Multi-Agent Dataset) for two-person interaction and makes it compatible with the ShelfAssembly dataset format. The dataset extracts motion data from multi-person interactions and outputs it in a standardized format suitable for motion generation and analysis tasks.

## Key Changes

The modified `comad.py` file has been updated to:

1. **Add ShelfAssembly-Compatible Class**: New `CoMaDDataset` class alongside original `CoMaD` for backward compatibility
2. **Standardize Output Structure**: Returns (motion_dict, annotation_dict) tuples compatible with ShelfAssembly
3. **Support Multi-Person Handling**: Treats Alice and Bob as separate "persons" similar to Main/Sub in ShelfAssembly
4. **Flexible Joint Extraction**: Extracts 9 key interaction joints with 3D coordinates
5. **Graceful Error Handling**: Handles missing dependencies and invalid data gracefully

## Output Format

### Motion Dictionary

Each motion clip returns joint position features for both interaction partners:

| Feature | Shape | Description |
|---------|-------|-------------|
| `alice_joints` | `(N_frames, 9, 3)` | Alice's 9 joint positions (3D coordinates) |
| `bob_joints` | `(N_frames, 9, 3)` | Bob's 9 joint positions (3D coordinates) |
| `pose` | `(N_frames, 27)` | Flattened Alice pose (9 joints × 3) |
| `other_pose` | `(N_frames, 27)` | Flattened Bob pose (9 joints × 3) |
| `task` | str | Task name identifier |
| `episode` | str | Episode identifier |
| `person_id` | str | Primary person (Alice or Bob) |

### Annotation Dictionary

| Field | Type | Description |
|-------|------|-------------|
| `no` | int | Clip index |
| `task` | str | Task name |
| `episode` | str | Episode identifier |
| `clip_idx` | int | Clip index within episode |
| `person_id` | str | Primary person identifier |
| `valid_length` | int | Number of valid frames |
| `dataset` | str | Dataset name ('CoMaD') |

## Extracted Joints

CoMaD extracts 9 key joints for interaction analysis:

1. **BackTop** - Upper back
2. **LShoulderBack** - Left shoulder (back)
3. **RShoulderBack** - Right shoulder (back)
4. **LElbowOut** - Left elbow
5. **RElbowOut** - Right elbow
6. **LWristOut** - Left wrist
7. **RWristOut** - Right wrist
8. **LHandOut** - Left hand
9. **RHandOut** - Right hand

## Usage Example

```python
from data_loaders.comad.comad import CoMaDDataset

# Initialize dataset
dataset = CoMaDDataset(
    input_n=15,              # Past frames
    output_n=15,             # Future frames
    sample_rate=120,         # Original Hz
    output_rate=15,          # Output Hz
    split='train',           # 'train', 'test', or 'val'
    data_dir="/path/to/comad/dataset",
    mapping_json="/path/to/joint_mapping.json"
)

# Get a single sample
motion, annotation = dataset[0]

# Access motion features
print(motion['alice_joints'].shape)  # (30, 9, 3)
print(motion['pose'].shape)          # (30, 27)

# Get annotation info
print(f"Task: {annotation['task']}")
print(f"Episode: {annotation['episode']}")
print(f"Primary person: {annotation['person_id']}")
```

## Integration with ShelfAssembly Pipeline

The CoMaD dataset can now be used interchangeably with ShelfAssembly in your pipeline:

```python
from data_loaders.shelf_assembly.dataset import ShelfAssemblyDataset
from data_loaders.comad.comad import CoMaDDataset
from torch.utils.data import DataLoader

# Both datasets have compatible output format
shelf_dataset = ShelfAssemblyDataset(mode='train')
comad_dataset = CoMaDDataset(split='train', data_dir="...")

# Create combined DataLoader
from torch.utils.data import ConcatDataset
combined = ConcatDataset([shelf_dataset, comad_dataset])
loader = DataLoader(combined, batch_size=32)

for motion, annotation in loader:
    # Both datasets produce compatible structures
    # motion has different keys but same principle
    person_pose = motion['pose'] if 'pose' in motion else motion['body_pose']
```

## Parameters

### Constructor Arguments

- `input_n` (int): Number of past frames to include (default: 15)
- `output_n` (int): Number of future frames to include (default: 15)
- `sample_rate` (int): Original sampling rate in Hz (default: 120)
- `output_rate` (int): Output sampling rate in Hz (default: 15)
- `split` (str): Dataset split - 'train', 'test', or 'val' (default: 'train')
- `data_dir` (str): Root path to CoMaD dataset
- `mapping_json` (str): Path to joint name to index mapping JSON file
- `**kwargs`: Additional arguments for future extensions

## Data Organization

The dataset expects the following structure:

```
data_dir/
├── {split}                           # train, test, or val
│   ├── {task}/
│   │   ├── HH/
│   │   │   ├── {episode}/
│   │   │   │   ├── data.json         # Motion data
│   │   │   │   ├── metadata.json     # Person names
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

## Key Features

1. **Multi-Person Interaction**: Handles two-person (Alice/Bob) interaction data
2. **Joint Position Extraction**: Extracts 9 key joints from full-body data
3. **Downsampling**: Converts from high-frequency (120 Hz) to output rate (15 Hz)
4. **Sliding Window Clips**: Creates overlapping motion clips for training
5. **Data Augmentation**: Creates flipped clips (Alice as primary, then Bob as primary)
6. **Missing Data Handling**: Automatically filters out sequences with missing values
7. **Backward Compatibility**: Original `CoMaD` class preserved for existing code

## Output Dimensions

### With default parameters (input_n=15, output_n=15):

```python
# Total sequence length: 30 frames
motion['alice_joints'].shape       # (30, 9, 3) = 270 values
motion['bob_joints'].shape         # (30, 9, 3) = 270 values
motion['pose'].shape               # (30, 27)
motion['other_pose'].shape         # (30, 27)

# Batched with batch_size=32:
motion['pose'].shape               # (32, 30, 27)
motion['other_pose'].shape         # (32, 30, 27)
```

## Notes

1. **Downsampling**: The dataset automatically downsamples from the original 120 Hz to the specified output rate
2. **Data Augmentation**: Each valid segment generates 2 clips (Alice primary, Bob primary), effectively doubling dataset size
3. **Joint Mapping**: The mapping JSON file specifies which indices in the raw data correspond to the named joints
4. **Missing Data**: Sequences with missing joint data are automatically filtered out during loading

## Related Files

- `comad.py`: Main implementation with both `CoMaD` (original) and `CoMaDDataset` (new)
- `dataset.py` (ShelfAssembly): Reference implementation for output format
- Joint mapping JSON: Configuration file for joint name to index mapping

## Compatibility

- **PyTorch**: Uses `torch.utils.data.Dataset` base class
- **NumPy**: Data processing
- **Hydra**: Configuration management (optional, with graceful fallback)
- **interact utilities**: Custom JSON reading utilities

## Backward Compatibility

- Original `CoMaD` class is preserved for existing code
- Returns 4 tensors: (alice_input, alice_output, bob_input, bob_output)
- Recommended to migrate to `CoMaDDataset` for new projects

## Future Extensions

Possible enhancements:
- RGB video frame loading
- Bone length/skeleton constraints
- Temporal augmentation (speed variations)
- Contact label prediction
- Task-conditioned generation
