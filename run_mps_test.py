#!/usr/bin/env python3
"""
Standalone script to run MPS tests without pytest conflicts
"""
import sys
import torch
import logging
from test_mps_proteinmpnn import TestProteinMPNNMPS, log_device_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StandaloneTester(TestProteinMPNNMPS):
    """Modified tester that doesn't use pytest fixtures"""
    
    def setup(self):
        """Setup test environment without pytest fixture decorator"""
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
        import platform
        self.logger.info(f"Platform: {platform.platform()}")
        self.logger.info(f"Processor: {platform.processor()}")

if __name__ == "__main__":
    print("="*60)
    print("RUNNING COMPREHENSIVE MPS TESTS FOR PROTEINMPNN")
    print("="*60)
    
    tester = StandaloneTester()
    tester.setup()
    
    # Run tests
    tests_to_run = [
        ("MPS Availability", tester.test_mps_availability),
        ("Device Initialization", tester.test_device_initialization),
        ("ProteinMPNN Model on MPS", tester.test_proteinmpnn_model_on_mps),
        ("Performance Comparison", tester.test_performance_comparison),
        ("Memory Transfer", tester.test_memory_transfer),
        ("Batch Size Scaling", tester.test_batch_size_scaling),
        ("ProteinMPNN Specific Operations", tester.test_proteinmpnn_specific_operations),
    ]
    
    for test_name, test_func in tests_to_run:
        print(f"\n>>> Running: {test_name}")
        try:
            test_func()
            print(f"✅ {test_name} passed")
        except Exception as e:
            if "skip" in str(e).lower():
                print(f"⏭️  {test_name} skipped: {e}")
            else:
                print(f"❌ {test_name} failed: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)