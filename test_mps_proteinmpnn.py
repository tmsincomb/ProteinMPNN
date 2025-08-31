#!/usr/bin/env python3
"""
Pytest suite for testing ProteinMPNN with Metal Performance Shaders (MPS) on Apple Silicon.
This test suite monitors GPU usage and verifies MPS acceleration.
"""

import pytest
import torch
import numpy as np
import time
import logging
import subprocess
import os
import sys
import json
import platform
import statistics
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add ProteinMPNN to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_mps_proteinmpnn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_gpu_memory_usage():
    """Get GPU memory usage on macOS using powermetrics (requires sudo)."""
    try:
        # Alternative method using ioreg that doesn't require sudo
        result = subprocess.run(
            ['ioreg', '-l', '-w', '0', '-r', '-c', 'IOAccelerator'],
            capture_output=True,
            text=True
        )
        return result.stdout
    except Exception as e:
        logger.warning(f"Could not get GPU memory stats: {e}")
        return None


def log_device_info(device):
    """Log detailed device information."""
    logger.info("="*60)
    logger.info("DEVICE INFORMATION")
    logger.info("="*60)
    
    if device.type == 'mps':
        logger.info(f"✅ Using Metal Performance Shaders (MPS)")
        logger.info(f"Device: {device}")
        logger.info(f"MPS is available: {torch.backends.mps.is_available()}")
        logger.info(f"MPS is built: {torch.backends.mps.is_built()}")
        
        # Test MPS with a simple operation
        test_tensor = torch.randn(100, 100, device=device)
        result = torch.matmul(test_tensor, test_tensor.T)
        logger.info(f"MPS test computation successful: tensor shape {result.shape}")
        
    elif device.type == 'cuda':
        logger.info(f"Using CUDA GPU: {device}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
    else:
        logger.info(f"Using CPU: {device}")
    
    logger.info("="*60)


class TestProteinMPNNMPS:
    """Test suite for ProteinMPNN MPS functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.logger = logger
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        log_device_info(self.device)
        
        # Log PyTorch version
        self.logger.info(f"PyTorch version: {torch.__version__}")
        
        # Log system info
        self.logger.info(f"Platform: {platform.platform()}")
        self.logger.info(f"Processor: {platform.processor()}")
        
    def test_mps_availability(self):
        """Test if MPS is available on the system."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST: MPS Availability")
        self.logger.info("="*60)
        
        is_apple_silicon = sys.platform == 'darwin' and 'arm' in platform.processor().lower()
        
        if is_apple_silicon:
            assert torch.backends.mps.is_built(), "MPS backend is not built in PyTorch"
            assert torch.backends.mps.is_available(), "MPS is not available on this Apple Silicon Mac"
            self.logger.info("✅ MPS is available and built correctly")
        else:
            self.logger.info("ℹ️  Not running on Apple Silicon, skipping MPS availability test")
            pytest.skip("Not running on Apple Silicon")
    
    def test_device_initialization(self):
        """Test the modified device initialization code."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST: Device Initialization")
        self.logger.info("="*60)
        
        # Simulate the device selection logic from protein_mpnn_run.py
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_name = "CUDA GPU"
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Apple Metal Performance Shaders (MPS)"
        else:
            device = torch.device("cpu")
            device_name = "CPU"
        
        self.logger.info(f"Device selected: {device_name}")
        self.logger.info(f"Device object: {device}")
        
        # Test tensor creation on selected device
        test_tensor = torch.randn(10, 10, device=device)
        assert test_tensor.device.type == device.type
        self.logger.info(f"✅ Successfully created tensor on {device_name}")
    
    def test_proteinmpnn_model_on_mps(self):
        """Test loading and running ProteinMPNN model on MPS."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST: ProteinMPNN Model on MPS")
        self.logger.info("="*60)
        
        try:
            from protein_mpnn_utils import ProteinMPNN
            
            # Create a small model for testing
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
            
            # Move model to device
            model.to(self.device)
            model.eval()
            
            self.logger.info(f"✅ Model successfully loaded on {self.device.type}")
            
            # Create dummy input
            batch_size = 1
            seq_len = 50
            
            X = torch.randn(batch_size, seq_len, 4, 3, device=self.device)
            S = torch.randint(0, 21, (batch_size, seq_len), device=self.device)
            mask = torch.ones(batch_size, seq_len, device=self.device)
            chain_M = torch.ones(batch_size, seq_len, device=self.device)
            residue_idx = torch.arange(seq_len, device=self.device).unsqueeze(0)
            chain_encoding_all = torch.zeros(batch_size, seq_len, device=self.device)
            randn_1 = torch.randn(batch_size, seq_len, device=self.device)
            
            # Measure inference time
            start_time = time.time()
            
            with torch.no_grad():
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn_1)
            
            inference_time = time.time() - start_time
            
            self.logger.info(f"✅ Model inference successful")
            self.logger.info(f"Output shape: {log_probs.shape}")
            self.logger.info(f"Inference time: {inference_time:.4f} seconds")
            self.logger.info(f"Device used: {log_probs.device}")
            
            assert log_probs.device.type == self.device.type
            
            # Log GPU memory if available
            if self.device.type == 'mps':
                gpu_info = get_gpu_memory_usage()
                if gpu_info:
                    self.logger.info("GPU Memory Info (subset):")
                    # Log first 500 chars of GPU info
                    self.logger.info(gpu_info[:500] + "..." if len(gpu_info) > 500 else gpu_info)
            
        except ImportError as e:
            self.logger.error(f"Could not import ProteinMPNN utils: {e}")
            pytest.skip("ProteinMPNN utils not available")
    
    def test_performance_comparison(self):
        """Compare performance between MPS and CPU."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST: Performance Comparison")
        self.logger.info("="*60)
        
        # Create test tensors
        size = 1000
        iterations = 100
        
        # Test on current device
        device_tensor = torch.randn(size, size, device=self.device)
        
        start_time = time.time()
        for _ in range(iterations):
            result = torch.matmul(device_tensor, device_tensor.T)
        
        if self.device.type == 'mps':
            # Ensure MPS operations complete
            torch.mps.synchronize()
        elif self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        device_time = time.time() - start_time
        
        # Test on CPU for comparison
        cpu_tensor = torch.randn(size, size, device='cpu')
        
        start_time = time.time()
        for _ in range(iterations):
            result_cpu = torch.matmul(cpu_tensor, cpu_tensor.T)
        cpu_time = time.time() - start_time
        
        self.logger.info(f"Matrix multiplication ({size}x{size}, {iterations} iterations):")
        self.logger.info(f"  {self.device.type.upper()} time: {device_time:.4f} seconds")
        self.logger.info(f"  CPU time: {cpu_time:.4f} seconds")
        
        if self.device.type in ['mps', 'cuda']:
            speedup = cpu_time / device_time
            self.logger.info(f"  Speedup: {speedup:.2f}x")
            
            if self.device.type == 'mps':
                assert speedup > 1.0, "MPS should be faster than CPU for large matrix operations"
                self.logger.info(f"✅ MPS provides {speedup:.2f}x speedup over CPU")
    
    def test_memory_transfer(self):
        """Test memory transfer between CPU and MPS."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST: Memory Transfer")
        self.logger.info("="*60)
        
        if self.device.type != 'mps':
            pytest.skip("Test only relevant for MPS")
        
        # Create CPU tensor
        cpu_tensor = torch.randn(1000, 1000)
        self.logger.info(f"Created CPU tensor: shape={cpu_tensor.shape}, device={cpu_tensor.device}")
        
        # Transfer to MPS
        start_time = time.time()
        mps_tensor = cpu_tensor.to('mps')
        transfer_to_time = time.time() - start_time
        self.logger.info(f"Transferred to MPS: device={mps_tensor.device}, time={transfer_to_time:.6f}s")
        
        # Perform computation on MPS
        result_mps = torch.matmul(mps_tensor, mps_tensor.T)
        self.logger.info(f"Computation on MPS complete: shape={result_mps.shape}")
        
        # Transfer back to CPU
        start_time = time.time()
        result_cpu = result_mps.to('cpu')
        transfer_from_time = time.time() - start_time
        self.logger.info(f"Transferred back to CPU: device={result_cpu.device}, time={transfer_from_time:.6f}s")
        
        assert mps_tensor.device.type == 'mps'
        assert result_cpu.device.type == 'cpu'
        self.logger.info("✅ Memory transfer test passed")
    
    def test_comprehensive_performance_comparison(self):
        """Comprehensive performance comparison between CPU and GPU for various operations."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST: Comprehensive Performance Comparison")
        self.logger.info("="*60)
        
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available, skipping comprehensive comparison")
        
        # Test various tensor sizes and operations
        test_configs = [
            {"name": "Small tensors (100x100)", "size": 100, "iterations": 1000},
            {"name": "Medium tensors (500x500)", "size": 500, "iterations": 200},
            {"name": "Large tensors (2000x2000)", "size": 2000, "iterations": 20},
            {"name": "Very large tensors (5000x5000)", "size": 5000, "iterations": 5},
        ]
        
        results = []
        
        for config in test_configs:
            self.logger.info(f"\nTesting: {config['name']}")
            size = config['size']
            iterations = config['iterations']
            
            # Benchmark on CPU
            cpu_times = self._benchmark_operations('cpu', size, iterations)
            
            # Benchmark on MPS
            mps_times = self._benchmark_operations('mps', size, iterations)
            
            # Calculate speedups
            speedups = {}
            for op in cpu_times:
                speedups[op] = cpu_times[op] / mps_times[op] if mps_times[op] > 0 else 0
            
            # Log results
            self.logger.info(f"  Results for {config['name']}:")
            for op in cpu_times:
                self.logger.info(f"    {op}:")
                self.logger.info(f"      CPU: {cpu_times[op]:.4f}s")
                self.logger.info(f"      MPS: {mps_times[op]:.4f}s")
                self.logger.info(f"      Speedup: {speedups[op]:.2f}x")
            
            results.append({
                'config': config,
                'cpu_times': cpu_times,
                'mps_times': mps_times,
                'speedups': speedups
            })
        
        # Generate summary report
        self._generate_performance_report(results)
    
    def _benchmark_operations(self, device: str, size: int, iterations: int) -> Dict[str, float]:
        """Benchmark various tensor operations on specified device."""
        times = {}
        
        # Matrix multiplication
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        start = time.time()
        for _ in range(iterations):
            c = torch.matmul(a, b)
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        times['matmul'] = time.time() - start
        
        # Element-wise operations
        start = time.time()
        for _ in range(iterations * 5):  # More iterations since these are faster
            c = a + b
            c = a * b
            c = torch.sin(a)
            c = torch.exp(a.clamp(-10, 10))  # Clamp to avoid overflow
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        times['element_wise'] = time.time() - start
        
        # Convolution (if size permits)
        if size >= 100:
            conv_size = min(size, 500)  # Limit size for memory
            x = torch.randn(1, 3, conv_size, conv_size, device=device)
            conv = torch.nn.Conv2d(3, 64, 3, device=device)
            
            start = time.time()
            for _ in range(max(1, iterations // 10)):
                y = conv(x)
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            times['convolution'] = time.time() - start
        
        # Batch matrix multiplication
        batch_size = min(10, iterations)
        a_batch = torch.randn(batch_size, size, size, device=device)
        b_batch = torch.randn(batch_size, size, size, device=device)
        
        start = time.time()
        for _ in range(max(1, iterations // batch_size)):
            c_batch = torch.bmm(a_batch, b_batch)
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        times['batch_matmul'] = time.time() - start
        
        # SVD (computationally intensive)
        if size <= 1000:  # Only for smaller sizes
            svd_matrix = torch.randn(size, size, device=device)
            start = time.time()
            for _ in range(max(1, iterations // 100)):
                u, s, v = torch.svd(svd_matrix)
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            times['svd'] = time.time() - start
        
        return times
    
    def _generate_performance_report(self, results: List[Dict]):
        """Generate a comprehensive performance report."""
        self.logger.info("\n" + "="*60)
        self.logger.info("PERFORMANCE SUMMARY REPORT")
        self.logger.info("="*60)
        
        # Calculate average speedups across all operations
        all_speedups = []
        for result in results:
            all_speedups.extend(result['speedups'].values())
        
        avg_speedup = statistics.mean(all_speedups)
        median_speedup = statistics.median(all_speedups)
        
        self.logger.info(f"\nOverall Performance Metrics:")
        self.logger.info(f"  Average MPS Speedup: {avg_speedup:.2f}x")
        self.logger.info(f"  Median MPS Speedup: {median_speedup:.2f}x")
        self.logger.info(f"  Min Speedup: {min(all_speedups):.2f}x")
        self.logger.info(f"  Max Speedup: {max(all_speedups):.2f}x")
        
        # Best and worst performing operations
        best_ops = []
        worst_ops = []
        
        for result in results:
            for op, speedup in result['speedups'].items():
                best_ops.append((op, speedup, result['config']['name']))
                worst_ops.append((op, speedup, result['config']['name']))
        
        best_ops.sort(key=lambda x: x[1], reverse=True)
        worst_ops.sort(key=lambda x: x[1])
        
        self.logger.info(f"\nTop 3 Best Performing Operations:")
        for op, speedup, config in best_ops[:3]:
            self.logger.info(f"  {op} ({config}): {speedup:.2f}x speedup")
        
        self.logger.info(f"\nTop 3 Worst Performing Operations:")
        for op, speedup, config in worst_ops[:3]:
            self.logger.info(f"  {op} ({config}): {speedup:.2f}x speedup")
        
        # Memory efficiency insights
        self.logger.info(f"\nMemory Transfer Insights:")
        self.logger.info(f"  Small tensors may have lower speedup due to transfer overhead")
        self.logger.info(f"  Large tensors show better GPU utilization")
        
        # Recommendations
        self.logger.info(f"\nRecommendations:")
        if avg_speedup > 2.0:
            self.logger.info(f"  ✅ MPS provides significant performance benefits")
            self.logger.info(f"  ✅ Recommended for production use with ProteinMPNN")
        elif avg_speedup > 1.5:
            self.logger.info(f"  ⚠️  MPS provides moderate performance benefits")
            self.logger.info(f"  ⚠️  Consider using for large batch sizes")
        else:
            self.logger.info(f"  ❌ MPS provides minimal performance benefits")
            self.logger.info(f"  ❌ CPU might be sufficient for your use case")
    
    def test_proteinmpnn_specific_operations(self):
        """Test ProteinMPNN-specific operations on CPU vs GPU."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST: ProteinMPNN-Specific Operations")
        self.logger.info("="*60)
        
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
        
        try:
            from protein_mpnn_utils import ProteinMPNN
            
            # Test different model sizes - keeping consistent with ProteinMPNN default dimensions
            model_configs = [
                {"name": "Small", "hidden_dim": 128, "layers": 2},
                {"name": "Medium", "hidden_dim": 128, "layers": 3},
                {"name": "Large", "hidden_dim": 128, "layers": 4},
            ]
            
            sequence_lengths = [50, 100, 200, 400, 600, 800, 1000]
            
            for model_config in model_configs:
                self.logger.info(f"\nTesting {model_config['name']} model:")
                
                # Create model
                model = ProteinMPNN(
                    ca_only=False,
                    num_letters=21,
                    node_features=128,
                    edge_features=128,
                    hidden_dim=model_config['hidden_dim'],
                    num_encoder_layers=model_config['layers'],
                    num_decoder_layers=model_config['layers'],
                    augment_eps=0.0,
                    k_neighbors=48
                )
                
                for seq_len in sequence_lengths:
                    self.logger.info(f"  Sequence length: {seq_len}")
                    
                    # Prepare inputs - test with batch size 4 for more realistic comparison
                    batch_size = 4
                    X = torch.randn(batch_size, seq_len, 4, 3)
                    S = torch.randint(0, 21, (batch_size, seq_len))
                    mask = torch.ones(batch_size, seq_len)
                    chain_M = torch.ones(batch_size, seq_len)
                    residue_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
                    chain_encoding_all = torch.zeros(batch_size, seq_len)
                    randn_1 = torch.randn(batch_size, seq_len)
                    
                    # Benchmark on CPU
                    model_cpu = model.to('cpu')
                    model_cpu.eval()
                    inputs_cpu = [x.to('cpu') for x in [X, S, mask, chain_M, residue_idx, chain_encoding_all, randn_1]]
                    
                    cpu_times = []
                    for _ in range(5):  # Multiple runs for averaging
                        start = time.time()
                        with torch.no_grad():
                            _ = model_cpu(*inputs_cpu)
                        cpu_times.append(time.time() - start)
                    
                    # Benchmark on MPS
                    model_mps = model.to('mps')
                    model_mps.eval()
                    inputs_mps = [x.to('mps') for x in [X, S, mask, chain_M, residue_idx, chain_encoding_all, randn_1]]
                    
                    mps_times = []
                    for _ in range(5):  # Multiple runs for averaging
                        start = time.time()
                        with torch.no_grad():
                            _ = model_mps(*inputs_mps)
                        torch.mps.synchronize()
                        mps_times.append(time.time() - start)
                    
                    avg_cpu_time = statistics.mean(cpu_times)
                    avg_mps_time = statistics.mean(mps_times)
                    speedup = avg_cpu_time / avg_mps_time
                    
                    self.logger.info(f"    CPU: {avg_cpu_time:.4f}s (±{statistics.stdev(cpu_times):.4f}s)")
                    self.logger.info(f"    MPS: {avg_mps_time:.4f}s (±{statistics.stdev(mps_times):.4f}s)")
                    self.logger.info(f"    Speedup: {speedup:.2f}x")
                    
        except ImportError as e:
            self.logger.error(f"Could not import ProteinMPNN utils: {e}")
            pytest.skip("ProteinMPNN utils not available")
    
    def test_batch_size_scaling(self):
        """Test how performance scales with different batch sizes."""
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST: Batch Size Scaling")
        self.logger.info("="*60)
        
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        size = 512  # Fixed tensor size
        
        cpu_times = []
        mps_times = []
        
        for batch_size in batch_sizes:
            self.logger.info(f"\nTesting batch size: {batch_size}")
            
            # CPU benchmark
            cpu_tensor = torch.randn(batch_size, size, size, device='cpu')
            start = time.time()
            for _ in range(10):
                _ = torch.bmm(cpu_tensor, cpu_tensor.transpose(-2, -1))
            cpu_time = time.time() - start
            cpu_times.append(cpu_time)
            
            # MPS benchmark
            mps_tensor = torch.randn(batch_size, size, size, device='mps')
            start = time.time()
            for _ in range(10):
                _ = torch.bmm(mps_tensor, mps_tensor.transpose(-2, -1))
            torch.mps.synchronize()
            mps_time = time.time() - start
            mps_times.append(mps_time)
            
            speedup = cpu_time / mps_time
            self.logger.info(f"  CPU: {cpu_time:.4f}s")
            self.logger.info(f"  MPS: {mps_time:.4f}s")
            self.logger.info(f"  Speedup: {speedup:.2f}x")
        
        # Analyze scaling efficiency
        self.logger.info("\nScaling Analysis:")
        for i in range(1, len(batch_sizes)):
            batch_ratio = batch_sizes[i] / batch_sizes[0]
            cpu_time_ratio = cpu_times[i] / cpu_times[0]
            mps_time_ratio = mps_times[i] / mps_times[0]
            
            self.logger.info(f"  Batch size {batch_sizes[0]} → {batch_sizes[i]}:")
            self.logger.info(f"    Expected scaling: {batch_ratio:.1f}x")
            self.logger.info(f"    CPU scaling: {cpu_time_ratio:.2f}x")
            self.logger.info(f"    MPS scaling: {mps_time_ratio:.2f}x")
            self.logger.info(f"    MPS efficiency: {(batch_ratio/mps_time_ratio)*100:.1f}%")


def test_mps_proteinmpnn_integration():
    """Integration test for ProteinMPNN with MPS using a real PDB file."""
    logger.info("\n" + "="*60)
    logger.info("INTEGRATION TEST: ProteinMPNN with MPS")
    logger.info("="*60)
    
    # Check if we have a sample PDB file
    sample_pdb = Path("vendor/ProteinMPNN/inputs/1BC8.pdb")
    if not sample_pdb.exists():
        logger.warning(f"Sample PDB not found at {sample_pdb}")
        pytest.skip("Sample PDB file not available")
    
    # Check for model weights
    model_weights = Path("vendor/ProteinMPNN/vanilla_model_weights/v_48_020.pt")
    if not model_weights.exists():
        logger.warning(f"Model weights not found at {model_weights}")
        pytest.skip("Model weights not available")
    
    # Run ProteinMPNN with MPS
    import subprocess
    
    cmd = [
        "python", "protein_mpnn_run.py",
        "--pdb_path", str(sample_pdb),
        "--num_seq_per_target", "1",
        "--sampling_temp", "0.1",
        "--out_folder", "test_output/",
        "--batch_size", "1"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd="vendor/ProteinMPNN",
        capture_output=True,
        text=True
    )
    
    logger.info("STDOUT:")
    logger.info(result.stdout)
    
    if result.stderr:
        logger.info("STDERR:")
        logger.info(result.stderr)
    
    # Check if MPS was used
    if "Using Apple Metal Performance Shaders (MPS)" in result.stdout:
        logger.info("✅ ProteinMPNN successfully used MPS!")
    elif "Using CUDA GPU" in result.stdout:
        logger.info("✅ ProteinMPNN successfully used CUDA GPU!")
    elif "Using CPU" in result.stdout:
        logger.info("⚠️  ProteinMPNN fell back to CPU")
    
    assert result.returncode == 0, f"ProteinMPNN failed with return code {result.returncode}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "--log-cli-level=INFO"])