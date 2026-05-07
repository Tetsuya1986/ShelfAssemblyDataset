# CORE4D Dataset Migration Guide

## Summary of Changes

The `dataset_hho.py` file has been refactored to provide a **ShelfAssembly-compatible interface** while preserving the original CORE4D data structure functionality.

## Key Modifications

### 1. Class Naming and Imports

**Before:**
```python
class Dataset(Dataset):
```

**After:**
```python
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d

class CORE4DDataset(Dataset):
```

**Rationale**: Clear naming prevents conflicts; rotation conversion utilities imported for 6D representation.

### 2. Initialization Simplification

**Before**: 
- Complex SMPLX model initialization
- Direct vertex/normal computation
- Hierarchical frame storage

**After**:
- Simplified initialization focusing on motion extraction
- Graceful error handling with try-except blocks
- Two-stage loading: motion_data (tensor format) and raw_data (reference)

```python
self.motion_data = []   # Motion clips with 6D rotations
self.raw_data = []      # Raw data for reference
self.idx2frame = []     # (seq_id, frame_idx, bias)
```

### 3. Motion Feature Extraction

**New Method: `_extract_motion_features()`**

Converts SMPL parameters to ShelfAssembly format:

```python
def _extract_motion_features(self, human_params, seq_idx, clip_name, seq_name, person_id):
    """
    Extracts and converts motion features to 6D rotation format.
    
    Output format:
    - global_orient: (N_frame, 6) ← converted from axis-angle
    - body_pose: (N_frame, 252) ← 42 joints × 6D
    - left_hand_pose: (N_frame, 60) ← 10 joints × 6D
    - right_hand_pose: (N_frame, 60) ← 10 joints × 6D
    - root_pos: (N_frame, 3) ← translation only
    """
```

**Key transformation**:
- Axis-angle → Rotation matrix → 6D representation
- Consistent with ShelfAssembly pipeline

### 4. Data Structure Changes

#### Before (Complex hierarchical structure):
```python
output = {
    "frames": [
        {
            'person1_params': {...},
            'person2_params': {...},
            'objfit_params': {...},
            'p1_pelvis': ...,
            'p1_verts': ...,
            'p1_contact_label': ...,
            ...
        }
    ],
    "centroid": ...,
    "rotation": ...,
    "obj_model_path": ...,
}
```

#### After (Flat, efficient format):
```python
motion = {
    'global_orient': torch.Tensor,      # (N_frames, 6)
    'body_pose': torch.Tensor,          # (N_frames, 252)
    'left_hand_pose': torch.Tensor,     # (N_frames, 60)
    'right_hand_pose': torch.Tensor,    # (N_frames, 60)
    'root_pos': torch.Tensor,           # (N_frames, 3)
    'clip_name': str,
    'seq_name': str,
    'person_id': str,
}

annotation = {
    'no': int,
    'clip_name': str,
    'seq_name': str,
    'person_id': str,
    'valid_length': int,
    'dataset': 'CORE4D'
}
```

### 5. __getitem__ Implementation

**Before**: Complex frame-by-frame canonicalization and transformation

**After**: Simple slicing with metadata preservation

```python
def __getitem__(self, idx):
    # Get motion clip from pre-processed data
    motion_clip = {}
    for key in ['global_orient', 'body_pose', ...]:
        motion_clip[key] = motion[key][start_frame:end_frame:self.sample_rate]
    
    # Return motion and annotation (ShelfAssembly format)
    return motion_clip, annotation
```

**Benefits**:
- ✅ Faster loading (no per-frame transforms)
- ✅ Consistent with ShelfAssembly
- ✅ Easier to extend with augmentations

### 6. Error Handling

**Improvements**:
- Graceful handling of missing train/test split files
- Try-except around SMPLX imports
- Warning messages for missing features

```python
try:
    train_sequence_names, ... = load_train_test_split()
except Exception as e:
    print(f"Warning: Could not load train/test split: {e}")
    train_sequence_names = []
```

## Output Format Comparison

| Aspect | Before | After |
|--------|--------|-------|
| Rotation format | Axis-angle (3D) | 6D (matrix columns) |
| Clip structure | Nested frames list | Flat tensors |
| Return value | Single complex dict | (motion, annotation) tuple |
| Loading speed | Slower (per-frame transform) | Faster (pre-computed) |
| ShelfAssembly compatible | ❌ No | ✅ Yes |

## Dimension Compatibility

### ShelfAssembly Output:
```
global_orient:  (N_frames, 6)
body_pose:      (N_frames, 252)
left_hand_pose: (N_frames, 60)
right_hand_pose: (N_frames, 60)
root_pos:       (N_frames, 3)
```

### CORE4D Output (Now):
```
global_orient:  (N_frames, 6)    ✅ SAME
body_pose:      (N_frames, 252)  ✅ SAME
left_hand_pose: (N_frames, 60)   ✅ SAME
right_hand_pose: (N_frames, 60)  ✅ SAME
root_pos:       (N_frames, 3)    ✅ SAME
```

## Migration Path for Existing Code

### If using original Dataset class:

**Before:**
```python
from data_loaders.core4d.dataset_hho import Dataset
dataset = Dataset(mode='train', past_len=15, future_len=15, dataset_root="...")
sample = dataset[0]
```

**After:**
```python
from data_loaders.core4d.dataset_hho import CORE4DDataset
dataset = CORE4DDataset(mode='train', past_len=15, future_len=15, dataset_root="...")
motion, annotation = dataset[0]
```

### Accessing the data:

**Before:**
```python
frames = sample["frames"]
for frame in frames:
    pose = frame['person1_params']['pose']  # Axis-angle
    trans = frame['person1_params']['trans']
```

**After:**
```python
global_orient = motion['global_orient']  # 6D rotation
body_pose = motion['body_pose']          # 6D rotation
root_pos = motion['root_pos']            # Translation
```

## Testing the New Implementation

Run the example at the bottom of `dataset_hho.py`:

```bash
cd /path/to/ShelfAssemblyDataset
python -m data_loaders.core4d.dataset_hho
```

Expected output:
```
global_orient shape: torch.Size([30, 6])
body_pose shape: torch.Size([30, 252])
left_hand_pose shape: torch.Size([30, 60])
right_hand_pose shape: torch.Size([30, 60])
root_pos shape: torch.Size([30, 3])
```

## Benefits of the New Implementation

1. **Interoperability**: Use CORE4D and ShelfAssembly datasets interchangeably
2. **Efficiency**: Faster data loading with pre-computed rotations
3. **Consistency**: Unified data format across datasets
4. **Maintainability**: Simpler code structure
5. **Extensibility**: Easy to add new features (augmentation, multi-modal, etc.)
6. **Compatibility**: Works with existing ShelfAssembly-based training pipelines

## Backward Compatibility Notes

⚠️ **Breaking Changes**:
- Class name changed from `Dataset` to `CORE4DDataset`
- Return format changed from single dict to (motion, annotation) tuple
- No per-frame transforms in output
- Rotation representation changed to 6D

✅ **Preserved**:
- Same object category filtering
- Train/test/validation split handling
- Past/future frame extraction logic
- Multi-person data support

## Future Enhancements

Potential improvements while maintaining compatibility:
- Camera features loading (like ShelfAssembly's headcam/envcam)
- Augmentation pipeline
- Batch preprocessing
- GPU acceleration for rotation conversions
- Integration with text annotations

## Questions?

Refer to [README.md](README.md) for usage examples and API documentation.
