#!/usr/bin/env python3
"""
Quick GPU performance test for the Gibbs Generalization experiments.
"""

import torch
import time
import numpy as np
from main import gpu_diagnostic

def test_gpu_performance():
    """Test GPU performance and optimization features."""
    print("GPU Performance Test")
    print("=" * 50)
    
    # Run diagnostic
    device, mixed_precision_available = gpu_diagnostic()
    
    if device == 'cpu':
        print("⚠️  No GPU available - training will be slow")
        return
    
    print(f"\n✅ Testing GPU performance...")
    
    # Test basic operations
    print("Testing basic tensor operations...")
    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)
    
    start_time = time.time()
    z = torch.mm(x, y) + torch.randn_like(x)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    print(f"Large matrix operations: {gpu_time:.3f}s")
    
    # Test neural network forward pass
    print("Testing neural network performance...")
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 1)
    ).to(device)
    
    batch_size = 128
    input_data = torch.randn(batch_size, 784, device=device)
    
    # Warmup
    for _ in range(10):
        _ = model(input_data)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        output = model(input_data)
    
    torch.cuda.synchronize()
    nn_time = time.time() - start_time
    
    print(f"Neural network (100 forward passes): {nn_time:.3f}s")
    print(f"Speed: {100 / nn_time:.1f} forward passes/second")
    
    # Test mixed precision
    if mixed_precision_available:
        print("Testing mixed precision...")
        from torch.cuda.amp import autocast, GradScaler
        
        scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        target = torch.randn(batch_size, 1, device=device)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(50):
            optimizer.zero_grad()
            with autocast():
                output = model(input_data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        torch.cuda.synchronize()
        mixed_time = time.time() - start_time
        print(f"Mixed precision training (50 steps): {mixed_time:.3f}s")
        print(f"Speed: {50 / mixed_time:.1f} training steps/second")
    
    # Memory test
    print(f"\nGPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f} MB")
    
    print("\n✅ GPU test completed successfully!")
    print("Your GPU setup looks good for fast training.")

if __name__ == "__main__":
    test_gpu_performance()
