#!/usr/bin/env python3
"""
Production training script for ProteinMPNN with full dataset.
Based on the original ProteinMPNN training parameters.
WARNING: This will take 3-5 days on a GPU, 6-10 days on MPS, 25+ days on CPU.
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add training directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

from utils import build_training_clusters, PDB_dataset, loader_pdb, get_pdbs, StructureDataset, StructureLoader, worker_init_fn
from model_utils import ProteinMPNN, get_std_opt, featurize, loss_nll

def main():
    parser = argparse.ArgumentParser(description='Production training of ProteinMPNN')
    parser.add_argument('-i', '--input', type=str, default='./pdb_2021aug02',
                       help='Path to training data (default: ./pdb_2021aug02)')
    parser.add_argument('-o', '--output', type=str, default='./production_output',
                       help='Output directory (default: ./production_output)')
    parser.add_argument('--model-type', type=str, choices=['v_48_002', 'v_48_010', 'v_48_020', 'v_48_030'],
                       default='v_48_020', help='Model variant to train (default: v_48_020)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs (default: 150)')
    parser.add_argument('--examples-per-epoch', type=int, default=1000000,
                       help='Number of examples per epoch (default: 1000000)')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Batch size in tokens (default: 10000)')
    parser.add_argument('--max-length', type=int, default=10000,
                       help='Maximum protein length (default: 10000)')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save model every N epochs (default: 10)')
    parser.add_argument('--reload-every', type=int, default=2,
                       help='Reload data every N epochs (default: 2)')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'],
                       default='auto', help='Device to use (default: auto)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training (CUDA only)')
    parser.add_argument('--gradient-clip', type=float, default=-1.0,
                       help='Gradient clipping norm (negative to disable, default: -1.0)')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Set random seed: {args.seed}")
    
    print("=" * 80)
    print("ProteinMPNN Production Training")
    print("=" * 80)
    print("WARNING: This training will take several days to complete!")
    print("Estimated time: 3-5 days (GPU), 6-10 days (MPS), 25+ days (CPU)")
    print("=" * 80)
    
    # Check input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist!")
        print("Please provide path to pdb_2021aug02 data with -i option")
        print("\nTo download the training data:")
        print("wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz")
        print("tar -xzf pdb_2021aug02.tar.gz")
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
    if len(pt_files) < 100000:
        print(f"Warning: Only {len(pt_files)} .pt files found. Full dataset should have ~140,000 files.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
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
    
    # Model-specific noise levels
    noise_levels = {
        'v_48_002': 0.02,
        'v_48_010': 0.10,
        'v_48_020': 0.20,
        'v_48_030': 0.30
    }
    backbone_noise = noise_levels[args.model_type]
    
    # Setup parameters
    params = {
        "LIST": f"{args.input}/list.csv",
        "VAL": f"{args.input}/valid_clusters.txt",
        "TEST": f"{args.input}/test_clusters.txt",
        "DIR": args.input,
        "DATCUT": "2030-Jan-01",
        "RESCUT": 3.5,  # Resolution cutoff
        "HOMO": 0.70    # Min seq identity for homo chains
    }
    
    # Save configuration
    config = {
        "model_type": args.model_type,
        "backbone_noise": backbone_noise,
        "epochs": args.epochs,
        "examples_per_epoch": args.examples_per_epoch,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": device_name,
        "mixed_precision": args.mixed_precision,
        "gradient_clip": args.gradient_clip,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "training_start": datetime.now().isoformat(),
        "data_path": args.input,
        "output_path": args.output
    }
    
    with open(os.path.join(args.output, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nConfiguration:")
    print(f"  Model type: {args.model_type}")
    print(f"  Backbone noise: {backbone_noise:.2f}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Examples per epoch: {args.examples_per_epoch:,}")
    print(f"  Batch size: {args.batch_size:,} tokens")
    print(f"  Max protein length: {args.max_length:,}")
    print(f"  Save every: {args.save_every} epochs")
    print(f"  Reload data every: {args.reload_every} epochs")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Output: {args.output}")
    
    # Load data clusters (full dataset, not debug mode)
    print("\nLoading data clusters (this may take a few minutes)...")
    train, valid, test = build_training_clusters(params, debug=False)  # debug=False for full data
    print(f"  Train clusters: {len(train):,}")
    print(f"  Valid clusters: {len(valid):,}")
    print(f"  Test clusters: {len(test):,}")
    
    # Setup data loaders
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=True, 
        num_workers=args.num_workers, worker_init_fn=worker_init_fn
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=1, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn
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
        augment_eps=backbone_noise
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer
    start_epoch = 0
    total_step = 0
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        total_step = checkpoint.get('total_step', 0)
        print(f"  Resuming from epoch {start_epoch}, step {total_step}")
    
    optimizer = get_std_opt(model.parameters(), 128, total_step)
    
    # Mixed precision scaler for CUDA
    scaler = None
    if args.mixed_precision and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("  Using mixed precision training (CUDA)")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Production Training")
    print("=" * 80)
    
    log_file = os.path.join(args.output, 'log.txt')
    if start_epoch == 0:
        with open(log_file, 'w') as f:
            f.write("Epoch\tTrain_Loss\tVal_Loss\tTrain_Perplexity\tVal_Perplexity\tTime\n")
    
    best_val_loss = float('inf')
    
    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Reload data periodically
        if epoch > 0 and epoch % args.reload_every == 0:
            print(f"Reloading training data...")
            train, valid, test = build_training_clusters(params, debug=False)
            train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=1, shuffle=True,
                num_workers=args.num_workers, worker_init_fn=worker_init_fn
            )
        
        # Training phase
        model.train()
        train_loss, train_count = 0., 0
        
        print(f"Sampling {args.examples_per_epoch:,} training examples...")
        train_data = get_pdbs(train_loader, 1, args.max_length, args.examples_per_epoch)
        
        if not train_data:
            print("Warning: No training data loaded!")
            continue
            
        dataset = StructureDataset(train_data, truncate=None, max_length=args.max_length)
        loader = StructureLoader(dataset, batch_size=args.batch_size)
        
        batch_count = 0
        for batch in loader:
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            
            optimizer.zero_grad()
            mask_for_loss = mask * chain_M
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av, _ = loss_nll(S, log_probs, mask_for_loss)
                
                # Backward pass with gradient scaling
                scaler.scale(loss_av).backward()
                
                # Gradient clipping if enabled
                if args.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward/backward pass
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                _, loss_av, _ = loss_nll(S, log_probs, mask_for_loss)
                loss_av.backward()
                
                # Gradient clipping if enabled
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                
                optimizer.step()
            
            train_loss += loss_av.item()
            train_count += 1
            batch_count += 1
            total_step += 1
            
            # Print progress every 100 batches
            if batch_count % 100 == 0:
                avg_loss = train_loss / train_count
                print(f"  Batch {batch_count}: loss = {loss_av.item():.4f}, avg = {avg_loss:.4f}")
        
        avg_train_loss = train_loss / max(train_count, 1)
        train_perplexity = np.exp(avg_train_loss)
        print(f"Training - Loss: {avg_train_loss:.4f}, Perplexity: {train_perplexity:.2f}")
        
        # Validation phase
        model.eval()
        val_loss, val_count = 0., 0
        
        # Use 5000 examples for validation
        val_examples = min(5000, len(valid_loader))
        print(f"Validating on {val_examples} examples...")
        
        with torch.no_grad():
            val_data = get_pdbs(valid_loader, 1, args.max_length, val_examples)
            if val_data:
                val_dataset = StructureDataset(val_data, truncate=None, max_length=args.max_length)
                val_loader = StructureLoader(val_dataset, batch_size=args.batch_size)
                
                for batch in val_loader:
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    mask_for_loss = mask * chain_M
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av, _ = loss_nll(S, log_probs, mask_for_loss)
                    val_loss += loss_av.item()
                    val_count += 1
        
        avg_val_loss = val_loss / max(val_count, 1)
        val_perplexity = np.exp(avg_val_loss)
        print(f"Validation - Loss: {avg_val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.output, 'model_weights', 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'total_step': total_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output, 'model_weights', f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'total_step': total_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint: epoch_{epoch+1}.pt")
        
        # Always save last checkpoint for resuming
        last_checkpoint_path = os.path.join(args.output, 'model_weights', 'last_checkpoint.pt')
        torch.save({
            'epoch': epoch + 1,
            'total_step': total_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'config': config
        }, last_checkpoint_path)
        
        # Log results
        epoch_time = time.time() - epoch_start
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t")
            f.write(f"{train_perplexity:.2f}\t{val_perplexity:.2f}\t{epoch_time:.1f}s\n")
        
        print(f"Epoch time: {epoch_time/60:.1f} minutes")
        
        # Estimate remaining time
        if epoch > start_epoch:
            avg_epoch_time = (time.time() - epoch_start) / (epoch - start_epoch + 1)
            remaining_epochs = args.epochs - epoch - 1
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_hours = eta_seconds / 3600
            eta_days = eta_hours / 24
            
            if eta_days > 1:
                print(f"Estimated time remaining: {eta_days:.1f} days")
            else:
                print(f"Estimated time remaining: {eta_hours:.1f} hours")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output}")
    print(f"Best model: {args.output}/model_weights/best_model.pt")
    print(f"Training log: {log_file}")
    
    # Save final summary
    summary = {
        "training_completed": datetime.now().isoformat(),
        "total_epochs": args.epochs,
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(avg_train_loss),
        "final_val_loss": float(avg_val_loss),
        "total_training_time": f"{(time.time() - epoch_start * args.epochs) / 3600:.1f} hours",
        "config": config
    }
    
    with open(os.path.join(args.output, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nTo use the trained model:")
    print(f"python protein_mpnn_run.py \\")
    print(f"    --pdb_path ./inputs/1QYS.pdb \\")
    print(f"    --pdb_path_chains A \\")
    print(f"    --out_folder ./inference_output/ \\")
    print(f"    --num_seq_per_target 10 \\")
    print(f"    --sampling_temp 0.1 \\")
    print(f"    --path_to_model_weights {args.output}/model_weights/ \\")
    print(f"    --model_name best_model")

if __name__ == "__main__":
    main()