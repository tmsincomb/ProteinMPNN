#!/usr/bin/env python3
"""
Comprehensive benchmark script for comparing CPU vs MPS performance on ProteinMPNN.
Tests with real PDB files and various batch sizes for realistic performance metrics.
"""

import torch
import numpy as np
import time
import json
import argparse
import statistics
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime

# Add ProteinMPNN to path
sys.path.insert(0, str(Path(__file__).parent))

from protein_mpnn_utils import parse_PDB, tied_featurize, ProteinMPNN
from protein_mpnn_run import featurize


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def benchmark_single_pdb(
    pdb_path: str,
    model: ProteinMPNN,
    device: str,
    batch_sizes: List[int],
    num_iterations: int = 5,
    num_sequences: int = 10
) -> Dict:
    """Benchmark a single PDB file with various batch sizes."""
    
    # Parse PDB
    pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    
    # Prepare features
    batch = []
    for _ in range(max(batch_sizes)):
        batch.append({
            'name': Path(pdb_path).stem,
            'seq_chain_A': '',
            'seq_chain_B': '',
            'seq_chain_C': '',
            'coords_chain_A': pdb_dict_list[0]['coords_chain_A'] if 'coords_chain_A' in pdb_dict_list[0] else None,
            'coords_chain_B': pdb_dict_list[0]['coords_chain_B'] if 'coords_chain_B' in pdb_dict_list[0] else None,
            'coords_chain_C': pdb_dict_list[0]['coords_chain_C'] if 'coords_chain_C' in pdb_dict_list[0] else None,
            'mask_chain_A': None,
            'mask_chain_B': None,
            'mask_chain_C': None,
        })
    
    # Get protein info
    all_coords = []
    for chain in ['A', 'B', 'C']:
        coords_key = f'coords_chain_{chain}'
        if coords_key in pdb_dict_list[0] and pdb_dict_list[0][coords_key] is not None:
            all_coords.append(pdb_dict_list[0][coords_key])
    
    if all_coords:
        total_length = sum(coords.shape[0] for coords in all_coords)
    else:
        # Fallback for different chain naming
        total_length = len(pdb_dict_list[0].get('seq', ''))
        if total_length == 0:
            # Try to get from first available coords
            for key in pdb_dict_list[0]:
                if 'coords' in key and pdb_dict_list[0][key] is not None:
                    total_length = pdb_dict_list[0][key].shape[0]
                    break
    
    results = {
        'pdb_name': Path(pdb_path).stem,
        'protein_length': total_length,
        'batch_results': []
    }
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    for batch_size in batch_sizes:
        print(f"\n  Testing batch size {batch_size} on {device}...")
        
        # Prepare batch
        current_batch = batch[:batch_size]
        
        try:
            # Featurize
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
                current_batch,
                device,
                None,  # chain_id_dict
                None,  # fixed_positions_dict
                [],    # omit_AAs_list
                None,  # tied_positions_dict
                None,  # pssm_dict
                None,  # bias_by_res_dict
                ca_only=False
            )
            
            # Add random noise for sampling
            randn_1 = torch.randn(chain_M.shape, device=device)
            
            # Memory before
            mem_before = get_memory_usage()
            
            # Warmup
            with torch.no_grad():
                _ = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn_1)
            
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            mem_during_max = mem_before
            
            for _ in range(num_iterations):
                start = time.time()
                
                with torch.no_grad():
                    for _ in range(num_sequences // batch_size):
                        log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn_1)
                
                if device == 'mps':
                    torch.mps.synchronize()
                elif device == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed = time.time() - start
                times.append(elapsed)
                
                # Track memory
                mem_current = get_memory_usage()
                mem_during_max = max(mem_during_max, mem_current)
            
            # Memory after
            mem_after = get_memory_usage()
            
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            
            results['batch_results'].append({
                'batch_size': batch_size,
                'device': device,
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min(times),
                'max_time': max(times),
                'sequences_per_second': (num_sequences / avg_time) if avg_time > 0 else 0,
                'memory_before_mb': mem_before,
                'memory_peak_mb': mem_during_max,
                'memory_after_mb': mem_after,
                'memory_used_mb': mem_during_max - mem_before
            })
            
        except Exception as e:
            print(f"    Error with batch size {batch_size} on {device}: {e}")
            results['batch_results'].append({
                'batch_size': batch_size,
                'device': device,
                'error': str(e)
            })
    
    return results


def run_comprehensive_benchmark():
    """Run comprehensive benchmarks on all available PDB files."""
    
    print("="*60)
    print("COMPREHENSIVE PROTEINMPNN CPU vs MPS BENCHMARK")
    print("="*60)
    
    # Check device availability
    if torch.backends.mps.is_available():
        devices = ['cpu', 'mps']
        print("✅ MPS is available - will compare CPU vs MPS")
    else:
        devices = ['cpu']
        print("ℹ️  MPS not available - CPU only benchmark")
    
    # Find PDB files
    pdb_paths = list(Path("inputs").glob("**/*.pdb"))
    print(f"\nFound {len(pdb_paths)} PDB files to benchmark")
    
    # Test configurations
    batch_sizes = [1, 2, 4, 8, 16]
    num_iterations = 3
    num_sequences = 10
    
    # Initialize model (using default settings)
    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=48
    )
    
    # Results storage
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'platform': sys.platform,
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'mps_available': torch.backends.mps.is_available(),
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3)
        },
        'benchmarks': []
    }
    
    # Run benchmarks for each PDB
    for pdb_path in pdb_paths[:3]:  # Limit to first 3 for faster testing
        print(f"\n\nBenchmarking {pdb_path.name}")
        print("-" * 40)
        
        for device in devices:
            print(f"\nDevice: {device.upper()}")
            results = benchmark_single_pdb(
                str(pdb_path),
                model,
                device,
                batch_sizes,
                num_iterations,
                num_sequences
            )
            all_results['benchmarks'].append(results)
    
    # Generate summary report
    generate_summary_report(all_results)
    
    # Save results
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


def generate_summary_report(results: Dict):
    """Generate a summary report of benchmark results."""
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY REPORT")
    print("="*60)
    
    # Collect all benchmark data
    cpu_times = []
    mps_times = []
    speedups = []
    
    for benchmark in results['benchmarks']:
        for batch_result in benchmark['batch_results']:
            if 'error' not in batch_result:
                if batch_result['device'] == 'cpu':
                    cpu_times.append(batch_result['avg_time'])
                elif batch_result['device'] == 'mps':
                    mps_times.append(batch_result['avg_time'])
    
    # Calculate speedups for matching configurations
    if cpu_times and mps_times and len(cpu_times) == len(mps_times):
        speedups = [cpu/mps for cpu, mps in zip(cpu_times, mps_times)]
        
        print(f"\nOverall Performance Metrics:")
        print(f"  Average CPU time: {statistics.mean(cpu_times):.4f}s")
        print(f"  Average MPS time: {statistics.mean(mps_times):.4f}s")
        print(f"  Average speedup: {statistics.mean(speedups):.2f}x")
        print(f"  Median speedup: {statistics.median(speedups):.2f}x")
        print(f"  Max speedup: {max(speedups):.2f}x")
        print(f"  Min speedup: {min(speedups):.2f}x")
    
    # Per-protein analysis
    print("\n\nPer-Protein Performance:")
    print("-" * 40)
    
    proteins_by_name = {}
    for benchmark in results['benchmarks']:
        name = benchmark['pdb_name']
        if name not in proteins_by_name:
            proteins_by_name[name] = {
                'length': benchmark['protein_length'],
                'cpu_results': [],
                'mps_results': []
            }
        
        for batch_result in benchmark['batch_results']:
            if 'error' not in batch_result:
                if batch_result['device'] == 'cpu':
                    proteins_by_name[name]['cpu_results'].append(batch_result)
                else:
                    proteins_by_name[name]['mps_results'].append(batch_result)
    
    for name, data in proteins_by_name.items():
        print(f"\n{name} (Length: {data['length']} residues):")
        
        if data['cpu_results'] and data['mps_results']:
            # Find matching batch sizes
            for cpu_res in data['cpu_results']:
                mps_res = next((m for m in data['mps_results'] 
                               if m['batch_size'] == cpu_res['batch_size']), None)
                if mps_res:
                    speedup = cpu_res['avg_time'] / mps_res['avg_time']
                    print(f"  Batch {cpu_res['batch_size']}: CPU={cpu_res['avg_time']:.4f}s, "
                          f"MPS={mps_res['avg_time']:.4f}s, Speedup={speedup:.2f}x")
                    print(f"    Memory: CPU={cpu_res['memory_used_mb']:.1f}MB, "
                          f"MPS={mps_res['memory_used_mb']:.1f}MB")
    
    print("\n" + "="*60)
    print("\nRecommendations:")
    if speedups and statistics.mean(speedups) > 2.0:
        print("  ✅ MPS provides significant speedup for ProteinMPNN")
        print("  ✅ Especially beneficial for batch processing and large proteins")
    elif speedups and statistics.mean(speedups) > 1.5:
        print("  ⚠️  MPS provides moderate speedup")
        print("  ⚠️  Consider using for large batch sizes and long sequences")
    else:
        print("  ℹ️  Performance gains vary by protein size and batch configuration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ProteinMPNN CPU vs MPS performance")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8],
                       help="Batch sizes to test")
    parser.add_argument("--num-iterations", type=int, default=3,
                       help="Number of iterations for each benchmark")
    parser.add_argument("--num-sequences", type=int, default=10,
                       help="Number of sequences to generate per benchmark")
    
    args = parser.parse_args()
    
    run_comprehensive_benchmark()