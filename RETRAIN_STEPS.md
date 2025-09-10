# ProteinMPNN Retraining Guide - Complete Steps

## Testing Small Set Training

This section provides exact steps to quickly test the training pipeline with your existing `pdb_2021aug02/` data using only the original ProteinMPNN training scripts.

### Quick Test Steps (2-5 minutes on MPS/GPU)

```bash
# 1. Navigate to ProteinMPNN training directory
cd /Users/tmsincomb/Scripps\ Research\ Dropbox/Troy\ Sincomb/repos/ForgeAb/vendor/ProteinMPNN/training

# 2. Run debug mode for quick test (100 examples, 2 epochs)
python training.py \
    --path_for_training_data ../pdb_2021aug02 \  # YOUR FOLDER: Existing PDB dataset
    --path_for_outputs ./test_run \              # WILL CREATE: Output folder
    --debug True \                                # Enables debug mode (minimal data)
    --num_epochs 2 \                              # PARAMETER: Just 2 epochs
    --num_examples_per_epoch 100 \                # PARAMETER: Only 100 examples
    --batch_size 1000 \                           # PARAMETER: Small batch size
    --save_model_every_n_epochs 1                 # Save after each epoch

# 3. Monitor output
# You should see:
# - "Using Apple Metal Performance Shaders (MPS)" on Mac
# - Training progress for each epoch
# - Loss values printed to console
# - Model saved to ./test_run/model_weights/

# 4. Verify the model was saved
ls -la ./test_run/model_weights/
# Should show: epoch_1.pt, epoch_2.pt

# 5. Check training log
cat ./test_run/log.txt
# Shows: Epoch  Train  Validation (tab-separated losses)
```

### Standard Small Training Run (10-15 minutes)

```bash
# For a more thorough but still quick test
cd /Users/tmsincomb/Scripps\ Research\ Dropbox/Troy\ Sincomb/repos/ForgeAb/vendor/ProteinMPNN/training

python training.py \
    --path_for_training_data ../pdb_2021aug02 \  # YOUR FOLDER: PDB dataset
    --path_for_outputs ./small_train \           # WILL CREATE: Output folder
    --num_epochs 5 \                              # PARAMETER: 5 epochs
    --num_examples_per_epoch 1000 \               # PARAMETER: 1000 examples per epoch
    --batch_size 5000 \                           # PARAMETER: Batch size in tokens
    --max_protein_length 5000 \                   # PARAMETER: Max protein length
    --save_model_every_n_epochs 2 \               # Save every 2 epochs
    --backbone_noise 0.20 \                       # PARAMETER: Standard noise level
    --num_neighbors 48                            # PARAMETER: K-nearest neighbors

# Expected output structure:
# ./small_train/
# ├── model_weights/
# │   ├── epoch_2.pt
# │   ├── epoch_4.pt
# │   └── epoch_5.pt
# └── log.txt
```

### Minimal Test with Custom Parameters

```bash
cd /Users/tmsincomb/Scripps\ Research\ Dropbox/Troy\ Sincomb/repos/ForgeAb/vendor/ProteinMPNN/training

# Ultra-fast test with tiny dataset
python training.py \
    --path_for_training_data ../pdb_2021aug02 \
    --path_for_outputs ./minimal_test \
    --debug True \
    --num_epochs 1 \                    # PARAMETER: Single epoch
    --num_examples_per_epoch 50 \       # PARAMETER: Only 50 examples
    --batch_size 500 \                  # PARAMETER: Very small batch
    --max_protein_length 1000 \         # PARAMETER: Short proteins only
    --save_model_every_n_epochs 1

# Runtime: ~1 minute on MPS
```

### Verifying Successful Training

```bash
# 1. Check that model files were created
ls -la training/test_run/model_weights/*.pt
# Should list epoch_1.pt, epoch_2.pt

# 2. Verify log file has training metrics
cat training/test_run/log.txt
# Example output:
# Epoch	Train	Validation
# 1	4.2851	4.3012
# 2	4.1923	4.2156

# 3. Test the trained model on a sample protein
cd ..  # Back to ProteinMPNN root
python protein_mpnn_run.py \
    --pdb_path ./inputs/1QYS.pdb \                                     # BUILT-IN: Sample PDB
    --pdb_path_chains A \
    --out_folder ./test_inference/ \                                   # WILL CREATE: Output folder
    --num_seq_per_target 2 \
    --sampling_temp "0.1" \
    --path_to_model_weights ./training/test_run/model_weights/ \       # FROM ABOVE: Your model
    --model_name epoch_2                                               # Specify which checkpoint

# 4. Check generated sequences
cat ./test_inference/*.fa
# Should show FASTA format with designed sequences
```

### Expected Output & Timing

| Training Mode | Examples | Epochs | Expected Time (MPS) | Output Files |
|--------------|----------|--------|-------------------|--------------|
| Debug mode | 100 | 2 | 1-2 minutes | 2 model files + log.txt |
| Small training | 1000 | 5 | 10-15 minutes | 2-3 model files + log.txt |
| Minimal test | 50 | 1 | ~1 minute | 1 model file + log.txt |
| Full training | 1000000 | 150 | 3-5 days | 15 model files + log.txt |

### Troubleshooting

If training fails:

```bash
# 1. Verify data exists and is readable
ls ../pdb_2021aug02/*.pt | wc -l
# Should show number of .pt files (e.g., 142689)

# 2. Check MPS/CUDA availability
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}, CUDA: {torch.cuda.is_available()}')"

# 3. Test with even smaller parameters
python training.py \
    --path_for_training_data ../pdb_2021aug02 \
    --path_for_outputs ./tiny_test \
    --debug True \
    --num_epochs 1 \
    --num_examples_per_epoch 10 \      # Only 10 examples
    --batch_size 100 \                  # Tiny batch
    --max_protein_length 500

# 4. Check memory usage
# On Mac: Open Activity Monitor > Memory tab
# Watch Python process during training
```

### Success Indicators

✅ Training is working correctly if:
- Console shows "Using Apple Metal Performance Shaders (MPS)" or "Using CUDA GPU"
- Training losses are printed for each epoch
- Validation losses are computed and shown
- Model files (epoch_N.pt) are created in model_weights/
- log.txt contains decreasing or stable loss values
- No error messages or crashes occur

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Preparation](#data-preparation)
4. [Training Configuration](#training-configuration)
5. [Running Training](#running-training)
6. [Model Checkpointing](#model-checkpointing)
7. [Validation & Testing](#validation-testing)
8. [MPS/GPU Optimization](#mpsgpu-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Strategies](#advanced-strategies)

## Overview

ProteinMPNN is a message-passing neural network designed for protein sequence design. Retraining allows customization for specific protein families, organisms, or design objectives.

### Key Architecture Components
- **Message Passing Layers**: 3 encoder + 3 decoder layers (default)
- **Hidden Dimension**: 128 (default)
- **K-Nearest Neighbors**: 48 (default for spatial graph construction)
- **Node Features**: Backbone coordinates, distances, angles
- **Training Objective**: Masked amino acid prediction (similar to BERT)

## Prerequisites

### 1. Environment Setup
```bash
# Create conda environment
conda create --name proteinmpnn_train python=3.9
conda activate proteinmpnn_train

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch

# For Apple Silicon (M1/M2/M3) with MPS support
conda install pytorch torchvision torchaudio -c pytorch

# Install additional dependencies
pip install numpy pandas tqdm scipy
pip install rich click  # Optional: for better CLI interface
```

### 2. Hardware Requirements
- **Minimum**: 16GB RAM, GPU with 8GB VRAM
- **Recommended**: 32GB+ RAM, GPU with 16GB+ VRAM (A100, V100, or M1 Max/Ultra)
- **Storage**: 20GB+ for training data, 5GB+ for checkpoints

## Data Preparation

### 1. Download Official Training Data
```bash
# Full training set (16.5 GB) - PDB biounits from August 2, 2021
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz
tar -xzf pdb_2021aug02.tar.gz

# Small test subset (47 MB) - for quick testing
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz
tar -xzf pdb_2021aug02_sample.tar.gz
```

### 2. Data Format Structure
```
pdb_2021aug02/
├── list.csv              # Master list of all chains
├── valid_clusters.txt    # Validation set clusters (30% seq identity)
├── test_clusters.txt     # Test set clusters
└── *.pt files           # PyTorch tensors with structure data
    ├── PDBID_CHAINID.pt # Individual chain data
    └── PDBID.pt         # Metadata and biounit info
```

### 3. Data Fields Explanation
**PDBID_CHAINID.pt contains:**
- `seq`: Amino acid sequence (string)
- `xyz`: Atomic coordinates [L,14,3]
- `mask`: Boolean mask for present atoms [L,14]
- `bfac`: Temperature factors [L,14]
- `occ`: Occupancy values [L,14]

**PDBID.pt contains:**
- `method`: Experimental method
- `resolution`: Structure resolution
- `tm`: TM-scores between chains
- `asmb_chains`: Biounit composition
- `asmb_xform`: Transformation matrices

### 4. Custom Dataset Preparation
```python
# Script to prepare custom PDB dataset
import torch
import numpy as np
from training.utils import parse_PDB  # Use ProteinMPNN's parser

def prepare_custom_dataset(pdb_files, output_dir):
    """Convert PDB files to ProteinMPNN training format"""
    
    for pdb_file in pdb_files:
        # Parse PDB
        coords, seq = parse_PDB(pdb_file)
        
        # Create tensor dict
        data = {
            'seq': seq,
            'xyz': coords,
            'mask': np.ones_like(coords[..., 0], dtype=bool),
            'bfac': np.zeros_like(coords[..., 0]),
            'occ': np.ones_like(coords[..., 0])
        }
        
        # Save as .pt file
        torch.save(data, f"{output_dir}/{pdb_name}.pt")
```

## Training Configuration

### 1. Standard Model Configurations
The official models use these settings:

| Model Name | Backbone Noise | K-Neighbors | Training Epochs |
|------------|---------------|-------------|-----------------|
| v_48_002   | 0.02 Å       | 48          | 150             |
| v_48_010   | 0.10 Å       | 48          | 150             |
| v_48_020   | 0.20 Å       | 48          | 150             |
| v_48_030   | 0.30 Å       | 48          | 150             |

### 2. Key Hyperparameters
```python
# Essential training parameters
params = {
    # Model architecture
    "hidden_dim": 128,           # Hidden dimension size
    "num_encoder_layers": 3,     # Encoder depth
    "num_decoder_layers": 3,     # Decoder depth
    "num_neighbors": 48,         # K for k-NN graph
    "dropout": 0.1,              # Dropout rate
    
    # Training dynamics
    "backbone_noise": 0.20,      # Coordinate noise (Angstroms)
    "batch_size": 10000,         # Tokens per batch (not proteins!)
    "num_epochs": 150,           # Total epochs
    "learning_rate": 0.0001,     # Initial LR
    
    # Data parameters
    "max_protein_length": 10000, # Maximum chain length
    "num_examples_per_epoch": 1000000,  # Examples per epoch
    "reload_data_every_n_epochs": 2,    # Data refresh frequency
    
    # Quality filters
    "rescut": 3.5,               # Resolution cutoff (Å)
    "min_seq_identity": 0.3,     # Cluster threshold
}
```

## Running Training

### 1. Basic Training Command
```bash
cd training/

python training.py \
    --path_for_training_data /path/to/pdb_2021aug02 \
    --path_for_outputs ./experiments/exp_001 \
    --num_epochs 150 \
    --batch_size 10000 \
    --backbone_noise 0.20 \
    --num_neighbors 48 \
    --hidden_dim 128 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --dropout 0.1 \
    --save_model_every_n_epochs 10
```

### 2. Resume from Checkpoint
```bash
python training.py \
    --path_for_training_data /path/to/pdb_2021aug02 \
    --path_for_outputs ./experiments/exp_001 \
    --previous_checkpoint ./experiments/exp_001/model_weights/epoch_50.pt \
    --num_epochs 150  # Will continue from checkpoint epoch
```

### 3. Debug Mode (Quick Testing)
```bash
python training.py \
    --path_for_training_data /path/to/pdb_2021aug02_sample \
    --path_for_outputs ./test_run \
    --debug True \
    --num_epochs 2 \
    --num_examples_per_epoch 100
```

### 4. Using the Streamlined Retrain Script (Already Implemented)

**Note: A streamlined `retrain.py` script with rich CLI interface already exists in this repository.**

The implemented `retrain.py` features:
- Automatic device detection (CUDA/MPS/CPU)
- Rich progress bars and formatted output
- Built-in test mode for MPS verification
- Checkpoint saving and resuming
- Comprehensive logging

```bash
# Basic usage with rich CLI interface
python retrain.py \
    --data-path /path/to/pdb_2021aug02 \    # YOUR FOLDER: PDB training dataset
    --output-path ./retrain_output \         # WILL CREATE: Output directory
    --model-type v_48_020 \                  # PARAMETER: Model variant
    --epochs 150 \
    --batch-size 10000 \
    --examples-per-epoch 1000000

# Test mode for quick MPS/GPU verification
python retrain.py \
    --data-path /path/to/pdb_2021aug02_sample \  # YOUR FOLDER: Sample dataset
    --output-path ./test_output \                 # WILL CREATE: Test output
    --test \                                       # Uses 1000 examples, 2 epochs max
    --gpu                                          # Auto-selects CUDA/MPS if available

# Resume from checkpoint
python retrain.py \
    --data-path /path/to/pdb_2021aug02 \
    --output-path ./retrain_output \
    --checkpoint ./retrain_output/model_weights/best_model_epoch_50.pt \  # FROM ABOVE: Previous run
    --epochs 150

# Debug mode with minimal data
python retrain.py \
    --data-path /path/to/pdb_2021aug02_sample \
    --output-path ./debug_output \
    --debug \
    --epochs 5
```

Key features of the implemented script:
- **Auto-detects device**: CUDA → MPS → CPU fallback
- **Rich progress display**: Live progress bars for epochs and batches
- **Smart checkpointing**: Saves best model and periodic checkpoints
- **Test mode**: Quick 1000-example test for MPS verification
- **Comprehensive logging**: Training metrics saved to `training.log`

## Model Checkpointing

### 1. Checkpoint Structure
```python
checkpoint = {
    'epoch': epoch,
    'step': total_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'validation_loss': val_loss,
    'best_val_loss': best_val_loss,
    'args': args  # Training configuration
}
```

### 2. Save/Load Functions
```python
# Save checkpoint
def save_checkpoint(model, optimizer, epoch, step, loss, path):
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

# Load checkpoint
def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['step']
```

## Validation & Testing

### 1. Validation During Training
The training script automatically validates on a held-out set:
```python
# Validation happens every epoch
# Metrics: perplexity, accuracy, loss
# Best model saved based on validation loss
```

### 2. Test Set Evaluation
```bash
# Run inference on test set
python test_inference.sh \
    --model_weights ./experiments/exp_001/model_weights/best_model.pt \
    --test_path /path/to/pdb_2021aug02/test_clusters.txt
```

### 3. Custom Evaluation Metrics
```python
def evaluate_model(model, test_loader, device):
    """Custom evaluation function"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # Forward pass
            output = model(batch)
            
            # Calculate metrics
            loss = criterion(output, target)
            accuracy = calculate_accuracy(output, target)
            
            total_loss += loss.item()
            total_accuracy += accuracy
    
    return total_loss / len(test_loader), total_accuracy / len(test_loader)
```

## MPS/GPU Optimization

### 1. Device Selection (with MPS Support)
```python
# Automatic device selection
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal Performance Shaders (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")
```

### 2. MPS-Specific Optimizations
```python
# For Apple Silicon (M1/M2/M3)
if device.type == 'mps':
    # MPS optimizations
    torch.mps.set_per_process_memory_fraction(0.9)  # Use 90% of available memory
    
    # Synchronize after operations
    torch.mps.synchronize()
    
    # Note: Mixed precision may not be fully supported
    # Use with caution or disable for MPS
```

### 3. Memory Management
```python
# Gradient accumulation for large proteins
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. Multi-GPU Training
```python
# DataParallel for multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM)
```bash
# Reduce batch size
--batch_size 5000  # Instead of 10000

# Reduce maximum protein length
--max_protein_length 5000  # Instead of 10000

# Enable gradient checkpointing (if implemented)
--gradient_checkpointing True
```

#### 2. Slow Training
```bash
# Increase number of data loader workers
--num_workers 8  # Default is 4

# Use mixed precision training (CUDA only)
--mixed_precision True

# Reduce validation frequency
--validate_every_n_epochs 5
```

#### 3. Poor Convergence
```bash
# Adjust learning rate
--learning_rate 0.0005  # Try different values

# Reduce backbone noise
--backbone_noise 0.1  # Instead of 0.2

# Increase batch size for more stable gradients
--batch_size 20000
```

#### 4. MPS-Specific Issues
```python
# If MPS operations fail, fallback to CPU for specific ops
try:
    result = torch.operation(tensor.to('mps'))
except:
    result = torch.operation(tensor.to('cpu')).to('mps')
```

## Advanced Strategies

### 1. Fine-Tuning for Specific Protein Families
```python
# Load pretrained model
model = ProteinMPNN()
checkpoint = torch.load('vanilla_model_weights/v_48_020.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze early layers
for param in model.encoder.parameters():
    param.requires_grad = False

# Train only decoder
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.00001  # Lower LR for fine-tuning
)
```

### 2. Custom Loss Functions
```python
def custom_loss(pred, target, weights=None):
    """Weighted loss for specific amino acids"""
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    
    if weights is not None:
        # Weight certain positions or amino acids
        ce_loss = ce_loss * weights
    
    return ce_loss.mean()
```

### 3. Data Augmentation
```python
def augment_structure(coords, noise_level=0.1):
    """Add noise to coordinates for robustness"""
    noise = torch.randn_like(coords) * noise_level
    return coords + noise

def rotate_structure(coords):
    """Random rotation augmentation"""
    R = random_rotation_matrix()
    return torch.matmul(coords, R)
```

### 4. Curriculum Learning
```python
# Start with easy examples, gradually increase difficulty
curriculum_schedule = {
    0: {"max_length": 100, "noise": 0.05},
    50: {"max_length": 300, "noise": 0.10},
    100: {"max_length": 500, "noise": 0.15},
    150: {"max_length": 1000, "noise": 0.20},
}

for epoch in range(num_epochs):
    if epoch in curriculum_schedule:
        update_training_params(curriculum_schedule[epoch])
```

### 5. Ensemble Training
```bash
# Train multiple models with different seeds
for seed in 42 111 222 333; do
    python training.py \
        --seed $seed \
        --path_for_outputs ./ensemble/model_$seed \
        --backbone_noise 0.20 \
        --num_epochs 150
done
```

### 6. Transfer Learning Pipeline
```python
# Step 1: Pretrain on general PDB data
pretrain_model(general_dataset)

# Step 2: Fine-tune on specific protein family
finetune_model(specific_dataset, pretrained_weights)

# Step 3: Task-specific adaptation
adapt_model(task_dataset, finetuned_weights)
```

## Performance Benchmarks

### Expected Training Times
| Hardware | Batch Size | Time per Epoch | Total Training (150 epochs) |
|----------|------------|----------------|----------------------------|
| V100 GPU | 10000 | ~45 min | ~5 days |
| A100 GPU | 10000 | ~30 min | ~3 days |
| M1 Max | 5000 | ~60 min | ~6 days |
| M2 Ultra | 10000 | ~40 min | ~4 days |
| CPU (32 cores) | 1000 | ~4 hours | ~25 days |

### Expected Model Performance
| Model | Sequence Recovery | Perplexity | Design Success Rate |
|-------|------------------|------------|-------------------|
| v_48_002 | 52.1% | 6.8 | ~65% |
| v_48_010 | 51.7% | 6.9 | ~63% |
| v_48_020 | 51.4% | 7.0 | ~62% |
| v_48_030 | 51.0% | 7.2 | ~60% |

## Best Practices

1. **Always validate on held-out data** - Use provided validation clusters
2. **Monitor for overfitting** - Track train vs validation loss
3. **Save checkpoints frequently** - Every 5-10 epochs minimum
4. **Use version control** - Track experiments with git
5. **Log everything** - Training configs, losses, hardware specs
6. **Test before full training** - Run debug mode first
7. **Use appropriate batch sizes** - Larger for GPU, smaller for MPS/CPU
8. **Consider computational cost** - Full training takes days on GPU

## Resources

- **Original Paper**: [Science 2022](https://www.science.org/doi/10.1126/science.add2187)
- **GitHub Repository**: https://github.com/dauparas/ProteinMPNN
- **Training Data**: https://files.ipd.uw.edu/pub/training_sets/
- **Model Weights**: Available in `vanilla_model_weights/`
- **Support**: GitHub Issues on the original repository

## Citation
```bibtex
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022}
}
```

## Notes for This Fork

This fork includes MPS support for Apple Silicon, making it possible to train on M1/M2/M3 Macs with reasonable performance. The training scripts have been updated to automatically detect and use MPS when available.

### MPS-Specific Considerations:
- Mixed precision training may have limited support
- Some operations may fallback to CPU automatically
- Performance is best with batch sizes 5000-10000
- Memory usage is more efficient than CUDA equivalent
- Synchronization calls may be needed after certain operations

---

**Last Updated**: September 2024
**Maintainer**: ProteinMPNN Fork with MPS Support