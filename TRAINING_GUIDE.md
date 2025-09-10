# ProteinMPNN Training Guide

## Quick Start

### Simplest Usage - Just Point and Train!

```bash
# Train with just the data path - everything else is auto-configured
python train_simple.py pdb_2021aug02

# Quick test to verify everything works (2 epochs, ~5 minutes)
python train_simple.py pdb_2021aug02 --test

# Train with custom output directory
python train_simple.py pdb_2021aug02 my_custom_model
```

## Advanced Training with Auto-Configuration

The new `retrain_v2.py` script intelligently detects your system and configures optimal settings:

```bash
# Minimal required arguments - auto-detects everything else
python retrain_v2.py --data-path pdb_2021aug02 --output-path ./my_model

# Quick test mode for verification
python retrain_v2.py --data-path pdb_2021aug02 --output-path ./test_model --test

# Resume from checkpoint
python retrain_v2.py --data-path pdb_2021aug02 --output-path ./my_model \
    --checkpoint ./my_model/model_weights/epoch_50.pt

# Override auto-detected settings if needed
python retrain_v2.py --data-path pdb_2021aug02 --output-path ./my_model \
    --batch-size 8000 --examples 50000
```

## Features of the New Training Scripts

### 🧠 Intelligent Auto-Detection

The script automatically detects and configures:

- **Hardware**: CUDA GPU, Apple MPS, or CPU
- **Memory**: Available GPU/System memory
- **Batch Size**: Optimal for your hardware (GPU: 2000-10000 tokens)
- **Workers**: Optimal number of data loading processes
- **Mixed Precision**: Enabled on compatible GPUs (CUDA 7.0+)
- **Examples per Epoch**: Based on device speed

### ✅ Complete Training Pipeline

Implements all critical components from the original ProteinMPNN training:

- **Label Smoothed Loss**: Uses `loss_smoothed()` for training
- **Mixed Precision**: Full GradScaler support for faster training
- **Background Data Loading**: ProcessPoolExecutor for efficient data pipeline
- **Periodic Data Reloading**: Every 2 epochs to prevent overfitting
- **Complete Checkpoints**: Includes all metadata (num_edges, noise_level)
- **Automatic Best Model Saving**: Tracks validation perplexity

### 📊 System Capabilities Display

```
System Capabilities
┌────────────────────────────┬───────────────────────────────────────┐
│ Device                     │ CUDA GPU (NVIDIA RTX 4090)           │
│ Memory                     │ 24.0 GB                               │
│ CPU Cores                  │ 32                                    │
│ Mixed Precision            │ ✓                                     │
│ Recommended Batch Size     │ 10,000 tokens                        │
│ Recommended Workers        │ 12                                    │
│ Recommended Examples/Epoch │ 100,000                               │
└────────────────────────────┴───────────────────────────────────────┘
```

## Hardware Requirements and Performance

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 16 GB minimum
- **Storage**: 50 GB for dataset + checkpoints

### Recommended Setup
- **GPU**: NVIDIA GPU with 8+ GB VRAM or Apple Silicon with MPS
- **RAM**: 32+ GB
- **Storage**: SSD with 100+ GB free

### Expected Training Times

| Hardware | Batch Size | Time per Epoch | 100 Epochs |
|----------|------------|----------------|------------|
| RTX 4090 | 10,000 | ~5 min | ~8 hours |
| RTX 3080 | 8,000 | ~8 min | ~13 hours |
| Apple M2 Max | 5,000 | ~15 min | ~25 hours |
| CPU (16 cores) | 1,000 | ~60 min | ~100 hours |

## Output Structure

```
my_model/
├── model_weights/
│   ├── best_model.pt         # Best validation perplexity
│   ├── epoch_10.pt           # Checkpoint every 10 epochs
│   ├── epoch_20.pt
│   └── ...
└── training.log              # Training metrics per epoch
```

## Checkpoint Format

All checkpoints include:
```python
{
    'epoch': int,              # Current epoch
    'step': int,               # Total training steps
    'num_edges': 48,           # k-NN graph edges
    'noise_level': float,      # Backbone noise (0.02/0.10/0.20/0.30)
    'model_state_dict': dict,  # Model weights
    'optimizer_state_dict': dict,  # Optimizer state
    'train_loss': float,       # Training loss
    'val_loss': float          # Validation loss
}
```

## Monitoring Training

Check the training log for progress:
```bash
tail -f my_model/training.log
```

Log format:
```
epoch  train_perplexity  val_perplexity  train_accuracy  val_accuracy  time
1      25.123           22.456          0.045           0.052         312.5s
2      20.234           18.123          0.087           0.095         298.3s
...
```

## Troubleshooting

### Out of Memory Errors

If you get OOM errors, the script will suggest reducing batch size:
```bash
# Manually set smaller batch size
python retrain_v2.py --data-path pdb_2021aug02 --output-path ./my_model \
    --batch-size 2000
```

### Slow Training on CPU

Force CPU usage (not recommended unless necessary):
```bash
python retrain_v2.py --data-path pdb_2021aug02 --output-path ./my_model \
    --force-cpu --batch-size 500 --examples 1000
```

### MPS Issues on Apple Silicon

If MPS fails, the script automatically falls back to CPU. You can also force CPU:
```bash
python retrain_v2.py --data-path pdb_2021aug02 --output-path ./my_model \
    --force-cpu
```

## Using Trained Models

After training, use your model for protein design:

```bash
# Use your trained model
python protein_mpnn_run.py \
    --model_weights ./my_model/model_weights/best_model.pt \
    --pdb_path ./input.pdb \
    --out_folder ./designs/
```

## Differences from Original Training Script

### Improvements in `retrain_v2.py`:
- ✅ **Auto-detection** of hardware capabilities
- ✅ **Simplified interface** - only need data and output paths
- ✅ **Intelligent defaults** based on your system
- ✅ **Better progress tracking** with rich library
- ✅ **Automatic mixed precision** on compatible GPUs
- ✅ **Graceful fallbacks** for unsupported features

### Maintains Critical Features:
- ✅ Label smoothed loss for training
- ✅ Background data loading with ProcessPoolExecutor  
- ✅ Periodic data reloading every 2 epochs
- ✅ Complete checkpoint metadata
- ✅ Same model architecture and parameters
- ✅ Compatible checkpoint format

## Advanced Options

### Custom Model Variants

The script auto-selects `v_48_020` (20% backbone noise) as the most robust default. To train other variants, modify the `detect_model_type()` function in `retrain_v2.py`.

### Multi-GPU Training

For multi-GPU setups, use the original `training/training.py` with distributed training support.

### Custom Data Augmentation

Modify the `backbone_noise` parameter in the model initialization to adjust coordinate perturbation during training.

## Questions or Issues?

1. Check system capabilities are detected correctly
2. Verify dataset has all required files (list.csv, valid_clusters.txt, etc.)
3. Ensure sufficient disk space for checkpoints
4. Try test mode first to verify setup

For additional help, refer to the original ProteinMPNN documentation or open an issue on GitHub.