#!/usr/bin/env python3
"""
Quick test script to verify ProteinMPNN training works with minimal data.
Tests training pipeline with 2 epochs and a very small subset of data.
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add training directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

from utils import build_training_clusters, PDB_dataset, loader_pdb, get_pdbs, StructureDataset, StructureLoader, worker_init_fn
from model_utils import ProteinMPNN, get_std_opt, featurize, loss_nll

def main():
    parser = argparse.ArgumentParser(description='Quick test of ProteinMPNN training')
    parser.add_argument('-i', '--input', type=str, default='./pdb_2021aug02',
                       help='Path to training data (default: ./pdb_2021aug02)')
    parser.add_argument('-o', '--output', type=str, default='./test_train_output',
                       help='Output directory (default: ./test_train_output)')
    parser.add_argument('--examples', type=int, default=50,
                       help='Number of examples to use (default: 50)')
    parser.add_argument('--batch-size', type=int, default=500,
                       help='Batch size in tokens (default: 500)')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'],
                       default='auto', help='Device to use (default: auto)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ProteinMPNN Quick Training Test")
    print("=" * 60)
    
    # Check input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist!")
        print("Please provide path to pdb_2021aug02 data with -i option")
        sys.exit(1)
    
    # Verify data files exist
    required_files = ['list.csv', 'valid_clusters.txt', 'test_clusters.txt']
    for f in required_files:
        if not os.path.exists(os.path.join(args.input, f)):
            print(f"Error: Required file {f} not found in {args.input}")
            sys.exit(1)
    
    # Count .pt files
    pt_files = list(Path(args.input).glob('*.pt'))
    print(f"Found {len(pt_files)} .pt files in {args.input}")
    if len(pt_files) < 100:
        print("Warning: Very few .pt files found. Training may not work properly.")
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_name = "CUDA GPU"
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Apple MPS"
        else:
            device = torch.device("cpu")
            device_name = "CPU"
    else:
        device = torch.device(args.device)
        device_name = args.device.upper()
    
    print(f"\nDevice: {device_name} ({device})")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'model_weights'), exist_ok=True)
    
    # Setup parameters
    params = {
        "LIST": f"{args.input}/list.csv",
        "VAL": f"{args.input}/valid_clusters.txt",
        "TEST": f"{args.input}/test_clusters.txt",
        "DIR": args.input,
        "DATCUT": "2030-Jan-01",
        "RESCUT": 3.5,
        "HOMO": 0.70
    }
    
    print("\nConfiguration:")
    print(f"  Examples per epoch: {args.examples}")
    print(f"  Batch size: {args.batch_size} tokens")
    print(f"  Epochs: 2")
    print(f"  Output: {args.output}")
    
    # Load data clusters
    print("\nLoading data clusters...")
    train, valid, test = build_training_clusters(params, debug=True)  # debug=True for minimal data
    print(f"  Train clusters: {len(train)}")
    print(f"  Valid clusters: {len(valid)}")
    
    # Setup data loaders
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=True, 
        num_workers=0, worker_init_fn=worker_init_fn
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=1, shuffle=True,
        num_workers=0, worker_init_fn=worker_init_fn
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = ProteinMPNN(
        node_features=128,
        edge_features=128, 
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        dropout=0.1,
        augment_eps=0.20  # backbone noise
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Setup optimizer
    optimizer = get_std_opt(model.parameters(), 128, 0)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    log_file = os.path.join(args.output, 'log.txt')
    with open(log_file, 'w') as f:
        f.write("Epoch\tTrain\tValidation\tTime\n")
    
    # Train for 2 epochs
    for epoch in range(2):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/2")
        
        # Training phase
        model.train()
        train_loss, train_count = 0., 0
        
        print(f"  Sampling {args.examples} training examples...")
        max_length = min(args.batch_size, 1000)  # max protein length
        
        # Sample training data
        train_data = get_pdbs(train_loader, 1, max_length, args.examples)
        if not train_data:
            print("  Warning: No training data loaded!")
            continue
            
        dataset = StructureDataset(train_data, truncate=None, max_length=max_length)
        loader = StructureLoader(dataset, batch_size=args.batch_size)
        
        batch_count = 0
        for batch in loader:
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            
            optimizer.zero_grad()
            mask_for_loss = mask * chain_M
            
            # Forward pass
            log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            _, loss_av, _ = loss_nll(S, log_probs, mask_for_loss)
            
            # Backward pass
            loss_av.backward()
            optimizer.step()
            
            train_loss += loss_av.item()
            train_count += 1
            batch_count += 1
            
            if batch_count % 5 == 0:
                print(f"    Batch {batch_count}: loss = {loss_av.item():.4f}")
        
        avg_train_loss = train_loss / max(train_count, 1)
        print(f"  Training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss, val_count = 0., 0
        
        print(f"  Validating on {min(20, len(valid_loader))} examples...")
        
        with torch.no_grad():
            val_data = get_pdbs(valid_loader, 1, max_length, min(20, len(valid_loader)))
            if val_data:
                val_dataset = StructureDataset(val_data, truncate=None, max_length=max_length)
                val_loader = StructureLoader(val_dataset, batch_size=args.batch_size)
                
                for batch in val_loader:
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    mask_for_loss = mask * chain_M
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av, _ = loss_nll(S, log_probs, mask_for_loss)
                    val_loss += loss_av.item()
                    val_count += 1
        
        avg_val_loss = val_loss / max(val_count, 1)
        print(f"  Validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output, 'model_weights', f'epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Log results
        epoch_time = time.time() - epoch_start
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{epoch_time:.1f}s\n")
        
        print(f"  Epoch time: {epoch_time:.1f} seconds")
    
    print("\n" + "=" * 60)
    print("Training Test Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {args.output}")
    print(f"Model weights: {args.output}/model_weights/")
    print(f"Training log: {log_file}")
    
    # Quick test of the model
    print("\nQuick inference test...")
    model.eval()
    with torch.no_grad():
        # Create dummy input
        dummy_X = torch.randn(1, 50, 4, 3).to(device)
        dummy_S = torch.randint(0, 21, (1, 50)).to(device)
        dummy_mask = torch.ones(1, 50).to(device)
        dummy_chain_M = torch.ones(1, 50).to(device)
        dummy_residue_idx = torch.arange(50).unsqueeze(0).to(device)
        dummy_chain_encoding = torch.zeros(1, 50).to(device)
        
        output = model(dummy_X, dummy_S, dummy_mask, dummy_chain_M, 
                      dummy_residue_idx, dummy_chain_encoding)
        print(f"  Model output shape: {output.shape}")
        print(f"  Expected shape: (1, 50, 21)")
        
        if output.shape == torch.Size([1, 50, 21]):
            print("  ✓ Model inference working correctly!")
        else:
            print("  ✗ Unexpected output shape")
    
    print("\nTo test the trained model on real proteins:")
    print(f"python protein_mpnn_run.py \\")
    print(f"    --pdb_path ./inputs/1QYS.pdb \\")
    print(f"    --pdb_path_chains A \\")
    print(f"    --out_folder ./test_output/ \\")
    print(f"    --num_seq_per_target 2 \\")
    print(f"    --sampling_temp 0.1 \\")
    print(f"    --path_to_model_weights {args.output}/model_weights/ \\")
    print(f"    --model_name epoch_2")

if __name__ == "__main__":
    main()