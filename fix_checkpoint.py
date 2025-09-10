#!/usr/bin/env python3
"""
Fix the trained checkpoint by adding missing keys required by protein_mpnn_run.py
"""

import torch
import sys
from pathlib import Path

def fix_checkpoint(checkpoint_path, output_path=None):
    """Fix a checkpoint by adding missing required keys."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Current keys:", list(checkpoint.keys()))
    
    # Add missing keys
    if 'noise_level' not in checkpoint:
        # Based on retrain.py, v_48_020 uses noise level 0.20
        checkpoint['noise_level'] = 0.20
        print("Added noise_level: 0.20")
    
    if 'num_edges' not in checkpoint:
        # Based on retrain.py, k_neighbors=48
        checkpoint['num_edges'] = 48
        print("Added num_edges: 48")
    
    # Save the fixed checkpoint
    if output_path is None:
        output_path = checkpoint_path
    
    torch.save(checkpoint, output_path)
    print(f"Fixed checkpoint saved to: {output_path}")
    print("New keys:", list(checkpoint.keys()))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_checkpoint.py <checkpoint_path> [output_path]")
        sys.exit(1)
    
    checkpoint_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    fix_checkpoint(checkpoint_path, output_path)