#!/usr/bin/env python3
"""Test script to diagnose MNIST loading issues."""

import torch
import torchvision
import os
import shutil

def test_mnist_loading():
    """Test MNIST dataset loading with detailed error reporting."""
    print("🔍 Testing MNIST dataset loading...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    
    data_root = './data'
    print(f"Data root: {os.path.abspath(data_root)}")
    
    # Check if MNIST directory exists
    mnist_path = os.path.join(data_root, 'MNIST')
    if os.path.exists(mnist_path):
        print(f"📁 MNIST directory exists at: {mnist_path}")
        raw_path = os.path.join(mnist_path, 'raw')
        if os.path.exists(raw_path):
            files = os.listdir(raw_path)
            print(f"📋 Raw files: {files}")
            # Check file sizes
            for file in files:
                file_path = os.path.join(raw_path, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"   {file}: {size:,} bytes")
    
    # Remove existing data to force fresh download
    if os.path.exists(mnist_path):
        print(f"🗑️  Removing existing MNIST data...")
        shutil.rmtree(mnist_path)
    
    try:
        print("⬇️  Downloading MNIST train dataset...")
        train_dataset = torchvision.datasets.MNIST(
            root=data_root, 
            train=True, 
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        print(f"✅ Train dataset loaded: {len(train_dataset)} samples")
        
        print("⬇️  Downloading MNIST test dataset...")
        test_dataset = torchvision.datasets.MNIST(
            root=data_root, 
            train=False, 
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
        
        # Test accessing a sample
        print("🔍 Testing data access...")
        sample_data, sample_label = train_dataset[0]
        print(f"✅ Sample shape: {sample_data.shape}, label: {sample_label}")
        
        print("🎉 MNIST dataset is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading MNIST: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mnist_loading()
    if not success:
        print("\n💡 Troubleshooting suggestions:")
        print("1. Check your internet connection")
        print("2. Try running: pip install --upgrade torch torchvision")
        print("3. Clear PyTorch cache: rm -rf ~/.cache/torch")
        print("4. Check disk space availability")
