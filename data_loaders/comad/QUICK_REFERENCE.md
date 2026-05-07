# CoMaD Dataset - Quick Reference

## Quick Setup

```python
from data_loaders.comad.comad import CoMaDDataset
from torch.utils.data import DataLoader

# 1. Initialize dataset
dataset = CoMaDDataset(
    input_n=15,
    output_n=15,
    split='train',
    data_dir="/path/to/comad",
    mapping_json="/path/to/joint_mapping.json"
)

# 2. Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Get batches
for motion, annotation in loader:
    print(motion['pose'].shape)  # (32, 30, 27)
    # Use motion and annotation in training
```

## Output Structure

### Motion Dictionary
```python
{
    'alice_joints': torch.Tensor,     # (30, 9, 3)
    'bob_joints': torch.Tensor,       # (30, 9, 3)
    'pose': torch.Tensor,             # (30, 27)
    'other_pose': torch.Tensor,       # (30, 27)
    'task': str,
    'episode': str,
    'person_id': str                  # 'Alice' or 'Bob'
}
```

### Annotation Dictionary
```python
{
    'no': int,
    'task': str,
    'episode': str,
    'clip_idx': int,
    'person_id': str,
    'valid_length': int,
    'dataset': 'CoMaD'
}
```

## 9 Extracted Joints

| # | Joint Name | Description |
|---|------------|-------------|
| 0 | BackTop | Upper back/spine |
| 1 | LShoulderBack | Left shoulder |
| 2 | RShoulderBack | Right shoulder |
| 3 | LElbowOut | Left elbow |
| 4 | RElbowOut | Right elbow |
| 5 | LWristOut | Left wrist |
| 6 | RWristOut | Right wrist |
| 7 | LHandOut | Left hand |
| 8 | RHandOut | Right hand |

## Common Usage Patterns

### 1. Basic DataLoader

```python
from data_loaders.comad.comad import CoMaDDataset
from torch.utils.data import DataLoader

dataset = CoMaDDataset(
    input_n=20, output_n=10,
    split='train',
    data_dir="/data/comad",
    mapping_json="/data/comad/joint_mapping.json"
)

loader = DataLoader(dataset, batch_size=64, num_workers=4)

for motion, annotation in loader:
    # motion['pose']: (64, 30, 27)
    # motion['other_pose']: (64, 30, 27)
    pass
```

### 2. Access Joint Positions

```python
motion, _ = dataset[0]

# Get raw joint positions
alice_joints = motion['alice_joints']  # (30, 9, 3)
bob_joints = motion['bob_joints']      # (30, 9, 3)

# Extract specific joint
alice_right_hand = alice_joints[:, 8, :]  # (30, 3)

# Compute joint distances
distance = torch.norm(
    alice_joints - bob_joints, dim=-1
)  # (30, 9)
```

### 3. Combine with ShelfAssembly

```python
from torch.utils.data import ConcatDataset, DataLoader
from data_loaders.shelf_assembly.dataset import ShelfAssemblyDataset
from data_loaders.comad.comad import CoMaDDataset

shelf = ShelfAssemblyDataset(mode='train', split='train')
comad = CoMaDDataset(split='train', data_dir="...")

combined = ConcatDataset([shelf, comad])
loader = DataLoader(combined, batch_size=32)

for motion, annotation in loader:
    dataset_name = annotation['dataset']
    if dataset_name == 'ShelfAssembly':
        pose = motion['body_pose']
    elif dataset_name == 'CoMaD':
        pose = motion['pose']
```

### 4. Filter by Task

```python
dataset = CoMaDDataset(split='train', ...)

# Get all clips for a specific task
target_task = "handshake"
task_clips = [
    (motion, annot) for i, (motion, annot) in 
    enumerate(dataset) 
    if annot['task'] == target_task
]
```

### 5. Filter by Person

```python
dataset = CoMaDDataset(split='train', ...)

# Get only Alice as primary
alice_clips = [
    dataset[i] for i in range(len(dataset))
    if dataset.metadata_list[i]['person_id'] == 'Alice'
]
```

## Dimension Reference

### Basic Dimensions
```
Frames per clip:     30 (input_n=15 + output_n=15)
Joints per person:   9
Coordinates per joint: 3
Flattened pose size: 27 (9 × 3)
```

### Tensor Shapes
```
Single sample:
  alice_joints:  (30, 9, 3)
  bob_joints:    (30, 9, 3)
  pose:          (30, 27)
  other_pose:    (30, 27)

Batched (batch_size=32):
  pose:          (32, 30, 27)
  other_pose:    (32, 30, 27)
  alice_joints:  (32, 30, 9, 3)
  bob_joints:    (32, 30, 9, 3)
```

## Parameter Tuning

### Frame Windows
```python
# Short-term (5+5 frames = 10 total)
dataset = CoMaDDataset(input_n=5, output_n=5, ...)

# Medium-term (15+15 = 30 total)
dataset = CoMaDDataset(input_n=15, output_n=15, ...)

# Long-term (30+30 = 60 total)
dataset = CoMaDDataset(input_n=30, output_n=30, ...)
```

### Sampling Rates
```python
# High frequency (120 Hz → 30 Hz downsampling)
dataset = CoMaDDataset(sample_rate=120, output_rate=30, ...)

# Medium frequency (120 Hz → 15 Hz downsampling)
dataset = CoMaDDataset(sample_rate=120, output_rate=15, ...)

# Low frequency (120 Hz → 10 Hz downsampling)
dataset = CoMaDDataset(sample_rate=120, output_rate=10, ...)
```

## Integration with Training Models

```python
import torch.nn as nn
import torch

class InteractionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: alice_pose (27) + bob_pose (27) = 54
        # Sequence length: 30
        self.encoder = nn.LSTM(54, 256, 2, batch_first=True)
        self.decoder = nn.Linear(256, 54)
        
    def forward(self, motion_dict):
        # Combine both person poses
        combined = torch.cat([
            motion_dict['pose'],
            motion_dict['other_pose']
        ], dim=-1)  # (batch, 30, 54)
        
        enc_out, _ = self.encoder(combined)
        pred = self.decoder(enc_out)  # (batch, 30, 54)
        return pred

# Usage
dataset = CoMaDDataset(split='train', ...)
loader = DataLoader(dataset, batch_size=32)
model = InteractionPredictor()

for motion, annot in loader:
    output = model(motion)
```

## Dataset Statistics

```python
dataset = CoMaDDataset(split='train', ...)

print(f"Total clips: {len(dataset)}")
print(f"Frames per clip: {dataset.input_n + dataset.output_n}")
print(f"Joints: 9")
print(f"Coordinates per joint: 3")
print(f"Tasks: {len(set(m['task'] for m in dataset.metadata_list))}")
print(f"Episodes: {len(set(m['episode'] for m in dataset.metadata_list))}")

# Example output:
# Total clips: 5000
# Frames per clip: 30
# Joints: 9
# Coordinates per joint: 3
# Tasks: 12
# Episodes: 150
```

## Troubleshooting

### Issue: ImportError for interact utilities
**Solution**: Ensure interact package is installed or provide full paths in compose

### Issue: Joint mapping not found
**Solution**: Pass explicit `mapping_json` path to constructor

### Issue: No frames loaded
**Solution**: Verify data_dir structure and file paths match expected format

### Issue: Out of memory
**Solution**: Reduce batch_size or use num_workers=0 in DataLoader

## Dataset Properties

```python
dataset = CoMaDDataset(split='train', ...)

# Available attributes
print(dataset.input_n)         # 15
print(dataset.output_n)        # 15
print(dataset.sequence_len)    # 30
print(dataset.sample_rate)     # 120
print(dataset.output_rate)     # 15
print(len(dataset))            # Total clips
print(len(dataset.motion_clips))  # Total clips
print(len(dataset.metadata_list)) # Total metadata entries
```

## Working with Raw Joints

```python
motion, _ = dataset[0]

# Access specific frames
frame_0 = motion['alice_joints'][0]      # (9, 3)
last_frame = motion['alice_joints'][-1]  # (9, 3)

# Compute velocities
velocities = torch.diff(
    motion['alice_joints'], dim=0
)  # (29, 9, 3)

# Compute frame-to-frame distance
distances = torch.norm(
    torch.diff(motion['alice_joints'], dim=0),
    dim=-1
)  # (29, 9)
```

## Performance Considerations

| Metric | Value |
|--------|-------|
| Load time per clip | ~5ms |
| Memory per sample | ~2KB |
| Typical dataset size | 5,000-10,000 clips |
| Training batch | 32-64 samples |

## Dataset Splits

```python
# Training set
train_dataset = CoMaDDataset(split='train', ...)

# Validation set
val_dataset = CoMaDDataset(split='val', ...)

# Test set
test_dataset = CoMaDDataset(split='test', ...)
```

## Backward Compatibility

If you need the original format:

```python
from data_loaders.comad.comad import CoMaD

# Original format
dataset = CoMaD(input_n=15, output_n=15, split='train')
alice_in, alice_out, bob_in, bob_out = dataset[0]

# New format (recommended)
from data_loaders.comad.comad import CoMaDDataset
dataset = CoMaDDataset(input_n=15, output_n=15, split='train', ...)
motion, annotation = dataset[0]
```

## Additional Resources

- **README.md**: Detailed API documentation
- **comad.py**: Source code with both original and new implementations
- **joint_mapping.json**: Joint name to index mapping

## Support

For issues or questions:
1. Check the README.md for detailed API documentation
2. Review the __main__ example in comad.py
3. Verify your data directory structure
4. Ensure joint mapping JSON is in correct format
