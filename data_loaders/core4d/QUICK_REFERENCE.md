# CORE4D Dataset - Quick Reference

## Installation & Setup

```python
# 1. Import the dataset class
from data_loaders.core4d.dataset_hho import CORE4DDataset

# 2. Initialize the dataset
dataset = CORE4DDataset(
    mode='train',                           # 'train' or 'test'
    past_len=15,                           # frames to look back
    future_len=15,                         # frames to predict
    sample_rate=1,                         # 1 = no subsampling
    dataset_root="/path/to/core4d",        # CORE4D data directory
    test_set="all"                         # for test mode: 'all', 'seen', or 'unseen'
)

# 3. Get a sample
motion, annotation = dataset[0]
```

## Output Structure

### Motion Features (Tensor Dictionary)
```python
motion = {
    'global_orient':   torch.Tensor,  # shape: (30, 6)  - 6D rotation
    'body_pose':       torch.Tensor,  # shape: (30, 252)- 42 joints × 6D
    'left_hand_pose':  torch.Tensor,  # shape: (30, 60) - 10 joints × 6D
    'right_hand_pose': torch.Tensor,  # shape: (30, 60) - 10 joints × 6D
    'root_pos':        torch.Tensor,  # shape: (30, 3)  - translation
    'clip_name':       'clip_001',
    'seq_name':        'sequence_01',
    'person_id':       'Person1',
    'no':              0
}
```

### Annotation Metadata
```python
annotation = {
    'no':            0,
    'clip_name':     'clip_001',
    'seq_name':      'sequence_01',
    'person_id':     'Person1',
    'valid_length':  30,
    'dataset':       'CORE4D'
}
```

## Common Usage Patterns

### 1. Basic DataLoader

```python
from torch.utils.data import DataLoader

dataset = CORE4DDataset(mode='train', dataset_root="...", past_len=20, future_len=10)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for motion_batch, annotation_batch in loader:
    # motion_batch is a dict of batched tensors
    # e.g., motion_batch['root_pos'].shape == (32, 30, 3)
    pass
```

### 2. Use with ShelfAssembly in Same Training Loop

```python
from data_loaders.shelf_assembly.dataset import ShelfAssemblyDataset
from data_loaders.core4d.dataset_hho import CORE4DDataset

# Both datasets now have compatible output format!
shelf_dataset = ShelfAssemblyDataset(mode='train', split='train')
core4d_dataset = CORE4DDataset(mode='train', dataset_root="...")

# Combine them
from torch.utils.data import ConcatDataset
combined = ConcatDataset([shelf_dataset, core4d_dataset])
loader = DataLoader(combined, batch_size=32)

for motion, annotation in loader:
    # Same format works for both!
    global_orient = motion['global_orient']  # (batch, time, 6)
    body_pose = motion['body_pose']          # (batch, time, 252)
```

### 3. Processing Motion Features

```python
# Convert 6D rotation back to other formats if needed
from utils.rotation_conversions import matrix_to_rotation_6d, axis_angle_to_matrix

motion, _ = dataset[0]

# Access features
global_orient_6d = motion['global_orient']  # (30, 6)
body_pose_6d = motion['body_pose']          # (30, 252)

# Stack all pose features
all_poses = torch.cat([
    global_orient_6d,
    body_pose_6d,
    motion['left_hand_pose'],
    motion['right_hand_pose']
], dim=-1)  # (30, 378)
```

### 4. Dataset Statistics

```python
dataset = CORE4DDataset(mode='train', dataset_root="...")

print(f"Total clips: {len(dataset)}")
print(f"Frames per clip: {dataset.past_len + dataset.future_len}")
print(f"Sample rate: {dataset.sample_rate}")

# Get sample and check shapes
motion, annot = dataset[0]
print(f"Motion features shape: {motion['global_orient'].shape}")
```

### 5. Multi-person Handling

```python
# The dataset creates separate clips for Person1 and Person2
dataset = CORE4DDataset(mode='train', dataset_root="...")
motion, annot = dataset[0]

person_id = annot['person_id']  # 'Person1' or 'Person2'
print(f"This motion is from: {person_id}")

# If you want to filter by person:
# Iterate through dataset and check annotation
for i in range(len(dataset)):
    motion, annot = dataset[i]
    if annot['person_id'] == 'Person1':
        # Process person 1 only
        pass
```

## Dimension Reference

### Rotation Representation
- **6D format**: First two columns of 3×3 rotation matrix
- **Input**: Axis-angle (3D) from SMPL
- **Output**: 6D for numerical stability

### Pose Hierarchy
```
Global Orientation:  1 joint   × 6 = 6D
Body Pose:          21 joints  × 6 = 126D (but structured as 252D for 42 body chains)
Left Hand:          10 joints  × 6 = 60D
Right Hand:         10 joints  × 6 = 60D
Total:              42 joints × 6 = 252D (body only)
```

### Expected Tensor Shapes
```
With past_len=15, future_len=15:
- All motion features: (30, feature_dim)
  - global_orient: (30, 6)
  - body_pose: (30, 252)
  - left_hand_pose: (30, 60)
  - right_hand_pose: (30, 60)
  - root_pos: (30, 3)

Batched (batch_size=32):
- All features: (32, 30, feature_dim)
```

## Parameter Tuning

### Frame Windows
```python
# Short-term motion (5 frames past, 5 future)
dataset = CORE4DDataset(past_len=5, future_len=5, sample_rate=1)  # 10 frames

# Medium-term motion (15 frames past, 15 future)
dataset = CORE4DDataset(past_len=15, future_len=15, sample_rate=1)  # 30 frames

# Long-term motion with subsampling
dataset = CORE4DDataset(past_len=30, future_len=30, sample_rate=2)  # 60 frames at 2x rate
```

## Integration with Training Models

```python
class MotionGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: all motion features concatenated
        # global_orient (6) + body_pose (252) + left_hand (60) + right_hand (60) + root_pos (3) = 381
        input_dim = 6 + 252 + 60 + 60 + 3
        
    def forward(self, motion_dict):
        # Concatenate features
        features = torch.cat([
            motion_dict['global_orient'],
            motion_dict['body_pose'],
            motion_dict['left_hand_pose'],
            motion_dict['right_hand_pose'],
            motion_dict['root_pos']
        ], dim=-1)  # (batch, time, 381)
        
        # Process...
        return features

# Usage
dataset = CORE4DDataset(mode='train', dataset_root="...")
loader = DataLoader(dataset, batch_size=32)
model = MotionGenerator()

for motion, annot in loader:
    output = model(motion)
```

## Dataset Properties

```python
dataset = CORE4DDataset(mode='train', dataset_root="...")

# Available attributes
print(dataset.past_len)      # 15
print(dataset.future_len)    # 15
print(dataset.sample_rate)   # 1
print(len(dataset))          # Total number of clips
print(len(dataset.seq_dirs)) # Number of sequences
```

## Troubleshooting

### Issue: "No valid motion clips found"
**Solution**: Check dataset_root path and ensure data.npz files exist

### Issue: ImportError for rotation_conversions
**Solution**: Ensure `utils/rotation_conversions.py` exists in project root

### Issue: Out of memory during loading
**Solution**: Reduce batch_size or set num_workers=0 in DataLoader

### Issue: Mismatched dimensions in model
**Solution**: Verify input expects (batch, time, features) format:
```python
# Correct order
motion_dict['global_orient'].shape  # (batch, time, 6)

# NOT
motion_dict['global_orient'].shape  # (time, batch, 6)  ❌
```

## Additional Resources

- **README.md**: Detailed API documentation
- **MIGRATION_GUIDE.md**: Changes from previous version
- **dataset_hho.py**: Source code with docstrings
- **shelf_assembly/dataset.py**: Reference implementation

## Support

For issues or questions:
1. Check the documentation in README.md
2. Review MIGRATION_GUIDE.md for compatibility notes
3. Examine the example code in dataset_hho.py's `__main__` section
4. Verify your CORE4D data structure matches expected format
