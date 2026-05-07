# CoMaD Dataset - Migration Guide

## Overview

This guide helps you migrate from the original `CoMaD` class to the new `CoMaDDataset` class that uses ShelfAssembly-compatible format.

## Key Differences

### Original CoMaD Class

```python
from data_loaders.comad.comad import CoMaD

dataset = CoMaD(
    input_n=15,
    output_n=15,
    split='train'
)

# Returns 4 separate tensors
alice_in, alice_out, bob_in, bob_out = dataset[0]

# Shapes:
# alice_in:   (15, 9, 3)
# alice_out:  (15, 9, 3)
# bob_in:     (15, 9, 3)
# bob_out:    (15, 9, 3)
```

### New CoMaDDataset Class

```python
from data_loaders.comad.comad import CoMaDDataset

dataset = CoMaDDataset(
    input_n=15,
    output_n=15,
    split='train',
    data_dir="/path/to/comad",
    mapping_json="/path/to/mapping.json"
)

# Returns (motion_dict, annotation_dict)
motion, annotation = dataset[0]

# motion contains:
# - alice_joints: (30, 9, 3)
# - bob_joints:   (30, 9, 3)
# - pose:         (30, 27)  # Flattened alice
# - other_pose:   (30, 27)  # Flattened bob

# annotation contains:
# - no, task, episode, clip_idx, person_id, valid_length, dataset
```

## Why Migrate?

### Advantages of New CoMaDDataset

1. **ShelfAssembly Compatibility**: Same output format as ShelfAssembly dataset
2. **DataLoader Ready**: Designed for PyTorch DataLoader batching
3. **Better Error Handling**: Graceful fallbacks for missing dependencies
4. **Mixed Dataset Support**: Can combine CoMaD with ShelfAssembly and CORE4D
5. **Rich Metadata**: Annotation dict provides full context for each clip
6. **Data Augmentation**: Built-in role flipping (Alice/Bob swapped)
7. **Missing Data Handling**: Automatic filtering of corrupted sequences

### What's Staying the Same

- Original `CoMaD` class still available for backward compatibility
- Same joint extraction (9 key joints)
- Same downsampling behavior
- Same data loading pipeline

## Migration Steps

### Step 1: Update Imports

**Before:**
```python
from data_loaders.comad.comad import CoMaD

dataset = CoMaD(input_n=15, output_n=15, split='train')
```

**After:**
```python
from data_loaders.comad.comad import CoMaDDataset

dataset = CoMaDDataset(
    input_n=15,
    output_n=15,
    split='train',
    data_dir="/path/to/comad",
    mapping_json="/path/to/joint_mapping.json"
)
```

### Step 2: Update Data Access

**Before:**
```python
for idx in range(len(dataset)):
    alice_in, alice_out, bob_in, bob_out = dataset[idx]
    
    # Use individual tensors
    combined_input = torch.cat([alice_in, bob_in], dim=-1)
    combined_output = torch.cat([alice_out, bob_out], dim=-1)
```

**After:**
```python
for idx in range(len(dataset)):
    motion, annotation = dataset[idx]
    
    # Use motion dictionaries
    combined_input = torch.cat([
        motion['pose'][:15],      # First 15 frames (input)
        motion['other_pose'][:15]
    ], dim=-1)
    
    combined_output = torch.cat([
        motion['pose'][15:],      # Last 15 frames (output)
        motion['other_pose'][15:]
    ], dim=-1)
```

### Step 3: Use DataLoader

**Before:**
```python
from torch.utils.data import DataLoader

dataset = CoMaD(input_n=15, output_n=15, split='train')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    alice_in, alice_out, bob_in, bob_out = batch
    # Shapes: (32, 15, 9, 3) for each
```

**After:**
```python
from torch.utils.data import DataLoader

dataset = CoMaDDataset(
    input_n=15,
    output_n=15,
    split='train',
    data_dir="..."
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for motion, annotation in loader:
    # motion['pose']: (32, 30, 27)
    # motion['other_pose']: (32, 30, 27)
    # annotation['task']: List of task names
```

### Step 4: Update Training Code

**Before:**
```python
class InteractionModel(nn.Module):
    def forward(self, alice_in, bob_in):
        # (batch, 15, 9, 3) each
        batch_size = alice_in.shape[0]
        
        # Flatten for processing
        alice_flat = alice_in.reshape(batch_size, 15, -1)  # (batch, 15, 27)
        bob_flat = bob_in.reshape(batch_size, 15, -1)
        
        combined = torch.cat([alice_flat, bob_flat], dim=-1)  # (batch, 15, 54)
        return combined

# Training loop
for alice_in, alice_out, bob_in, bob_out in loader:
    input_data = torch.cat([alice_in, bob_in], dim=-1)  # (batch, 15, 54)
    output_data = torch.cat([alice_out, bob_out], dim=-1)  # (batch, 15, 54)
    
    prediction = model(alice_in, bob_in)
```

**After:**
```python
class InteractionModel(nn.Module):
    def forward(self, motion):
        # motion['pose']: (batch, 30, 27)
        # motion['other_pose']: (batch, 30, 27)
        
        # Combine both persons
        combined = torch.cat([
            motion['pose'],
            motion['other_pose']
        ], dim=-1)  # (batch, 30, 54)
        
        return combined

# Training loop
for motion, annotation in loader:
    # Input: first 15 frames
    input_data = torch.cat([
        motion['pose'][:, :15, :],
        motion['other_pose'][:, :15, :]
    ], dim=-1)  # (batch, 15, 54)
    
    # Output: last 15 frames
    output_data = torch.cat([
        motion['pose'][:, 15:, :],
        motion['other_pose'][:, 15:, :]
    ], dim=-1)  # (batch, 15, 54)
    
    prediction = model(motion)
```

### Step 5: Handle Multi-Person

**Before:**
```python
# Original only distinguishes through flipping in data loading
# You had to track person identity manually
clip_1_alice_in, _, clip_1_bob_in, _ = dataset[0]
clip_2_alice_in, _, clip_2_bob_in, _ = dataset[1]  # Could be flipped version
```

**After:**
```python
# New class tracks person identity explicitly
motion_1, annot_1 = dataset[0]
motion_2, annot_2 = dataset[1]

if annot_1['person_id'] == 'Alice':
    # Alice is primary in first clip
    alice_pose = motion_1['pose']
    bob_pose = motion_1['other_pose']
elif annot_1['person_id'] == 'Bob':
    # Bob is primary in first clip
    alice_pose = motion_1['bob_joints'].reshape(-1, 27)
    bob_pose = motion_1['alice_joints'].reshape(-1, 27)
```

## Conversion Matrix

| Original | New | Notes |
|----------|-----|-------|
| `CoMaD()` | `CoMaDDataset()` | Add `data_dir` and `mapping_json` |
| `dataset[i]` returns 4 tensors | `dataset[i]` returns (motion_dict, annotation_dict) | Requires unpacking change |
| No metadata | Full annotation dict | Includes task, episode, person_id |
| Separate input/output | Combined sequence | Use slice to separate: `motion['pose'][:15]` vs `[15:]` |
| Generic person tracking | person_id field | 'Alice' or 'Bob' explicitly |
| Plain numpy arrays | PyTorch tensors | Automatic torch conversion |

## Full Example: Before and After

### Before: Complete Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loaders.comad.comad import CoMaD

# Dataset
dataset = CoMaD(input_n=15, output_n=15, split='train')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
class PredictionModel(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(54, 128, 2, batch_first=True)
        self.linear = nn.Linear(128, 54)
    
    def forward(self, alice, bob):
        # alice, bob: (batch, 15, 9, 3)
        batch_size = alice.shape[0]
        alice_flat = alice.reshape(batch_size, 15, 27)
        bob_flat = bob.reshape(batch_size, 15, 27)
        combined = torch.cat([alice_flat, bob_flat], dim=-1)  # (batch, 15, 54)
        
        lstm_out, _ = self.lstm(combined)
        output = self.linear(lstm_out)
        return output

model = PredictionModel()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Training
for epoch in range(10):
    for alice_in, alice_out, bob_in, bob_out in loader:
        # Prepare input/output
        input_alice = alice_in  # (batch, 15, 9, 3)
        input_bob = bob_in      # (batch, 15, 9, 3)
        
        target_alice = alice_out  # (batch, 15, 9, 3)
        target_bob = bob_out      # (batch, 15, 9, 3)
        
        target = torch.cat([
            target_alice.reshape(target_alice.shape[0], 15, 27),
            target_bob.reshape(target_bob.shape[0], 15, 27)
        ], dim=-1)  # (batch, 15, 54)
        
        # Forward pass
        pred = model(input_alice, input_bob)
        loss = criterion(pred, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### After: Complete Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loaders.comad.comad import CoMaDDataset

# Dataset
dataset = CoMaDDataset(
    input_n=15,
    output_n=15,
    split='train',
    data_dir="/path/to/comad",
    mapping_json="/path/to/mapping.json"
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model (identical, just different interface)
class PredictionModel(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(54, 128, 2, batch_first=True)
        self.linear = nn.Linear(128, 54)
    
    def forward(self, motion):
        # motion['pose']: (batch, 30, 27)
        # motion['other_pose']: (batch, 30, 27)
        
        # Combine
        combined = torch.cat([
            motion['pose'],
            motion['other_pose']
        ], dim=-1)  # (batch, 30, 54)
        
        lstm_out, _ = self.lstm(combined)
        output = self.linear(lstm_out)
        return output

model = PredictionModel()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Training
for epoch in range(10):
    for motion, annotation in loader:
        # Separate input and output windows
        input_data = torch.cat([
            motion['pose'][:, :15, :],
            motion['other_pose'][:, :15, :]
        ], dim=-1)  # (batch, 15, 54)
        
        target_data = torch.cat([
            motion['pose'][:, 15:, :],
            motion['other_pose'][:, 15:, :]
        ], dim=-1)  # (batch, 15, 54)
        
        # Forward pass
        pred = model(motion)
        loss = criterion(pred, target_data)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Can also access annotation for logging
        tasks = annotation['task']
        people = annotation['person_id']
```

## Common Issues and Solutions

### Issue 1: "AttributeError: 'CoMaDDataset' object has no attribute 'alice_input'"

**Problem**: Trying to use old attribute names on new class

**Solution**: Use new access pattern
```python
# Old way (won't work)
clip = dataset.alice_input[0]

# New way
motion, annotation = dataset[0]
alice_pose = motion['pose']
```

### Issue 2: Dimension mismatches in model

**Problem**: Model expects input shape (batch, 15, 54) but gets (batch, 30, 54)

**Solution**: Explicitly slice before/after frames
```python
# Before
input_seq = motion['pose'][:, :15, :]  # Get first 15 frames only
output_seq = motion['pose'][:, 15:, :]  # Get last 15 frames

# Pass separately if needed
```

### Issue 3: Missing annotation information

**Problem**: Old code doesn't track which clip is Alice/Bob primary

**Solution**: Use annotation dict
```python
motion, annotation = dataset[0]
print(f"Primary person: {annotation['person_id']}")
print(f"Task: {annotation['task']}")
print(f"Episode: {annotation['episode']}")
```

### Issue 4: Configuration loading error

**Problem**: Hydra compose fails but code still needs to run

**Solution**: Provide explicit paths
```python
# This will fail if compose doesn't work
dataset = CoMaDDataset(split='train')

# This will always work
dataset = CoMaDDataset(
    split='train',
    data_dir="/absolute/path/to/comad",
    mapping_json="/absolute/path/to/mapping.json"
)
```

## Rollback Plan

If you need to revert to original behavior:

```python
# Use original CoMaD class
from data_loaders.comad.comad import CoMaD

dataset = CoMaD(input_n=15, output_n=15, split='train')
alice_in, alice_out, bob_in, bob_out = dataset[0]
```

The original class is preserved for full backward compatibility.

## Timeline for Migration

1. **Immediate**: Update imports and data access patterns
2. **Phase 1**: Update training/evaluation scripts
3. **Phase 2**: Test with DataLoader integration
4. **Phase 3**: Verify compatibility with downstream tasks
5. **Phase 4**: Deprecate old patterns (optional)

## Support Resources

1. **README.md** - Detailed API documentation
2. **QUICK_REFERENCE.md** - Common patterns and examples
3. **comad.py** - Source code with inline documentation
4. **ShelfAssembly Reference** - See `/data_loaders/shelf_assembly/dataset.py`

## Conclusion

The migration from `CoMaD` to `CoMaDDataset` provides:
- ✅ Better data loading pipeline
- ✅ Backward compatibility (original class preserved)
- ✅ ShelfAssembly format compatibility
- ✅ Enhanced metadata tracking
- ✅ Smoother integration with modern training pipelines

Most changes are localized to:
- Data loading code
- Sample unpacking
- Input preparation before models

The models themselves can often be used unchanged with minimal interface adaptation.
