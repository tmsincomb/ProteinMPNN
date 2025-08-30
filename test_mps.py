#!/usr/bin/env python3
"""Test MPS availability for ProteinMPNN on Apple Silicon"""

import sys

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("MPS available:", torch.backends.mps.is_available())
    print("MPS built:", torch.backends.mps.is_built())
    
    if torch.backends.mps.is_available():
        # Test creating a tensor on MPS
        device = torch.device("mps")
        x = torch.randn(3, 3, device=device)
        print(f"Successfully created tensor on MPS: {x.device}")
        print("MPS is working! ProteinMPNN will use GPU acceleration.")
    else:
        print("\nMPS is not available. ProteinMPNN will fall back to CPU.")
        print("To enable MPS, ensure you have:")
        print("1. macOS 12.3+ (Monterey or later)")
        print("2. PyTorch 1.12+ with MPS support")
        print("   Install with: pip3 install torch torchvision torchaudio")
        
except ImportError:
    print("PyTorch is not installed.")
    print("\nTo install PyTorch with MPS support on Apple Silicon:")
    print("pip3 install torch torchvision torchaudio")
    print("\nOr using conda:")
    print("conda install pytorch torchvision torchaudio -c pytorch")
    sys.exit(1)