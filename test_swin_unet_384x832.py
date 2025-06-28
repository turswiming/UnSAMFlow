#!/usr/bin/env python3
"""
Test script for Swin-UNet with 384x832 input size
"""

import torch
import torch.nn as nn
from models.swin_unet import SwinUNet, SwinMaskUNet
from models.get_model import get_model, get_mask_model
from utils.config_parser import init_config
import time

def test_384x832_swin_unet():
    """Test Swin-UNet with 384x832 input"""
    print("=" * 60)
    print("Testing Swin-UNet with 384x832 input size")
    print("=" * 60)
    
    # Create Swin-UNet with 384x832 input
    model = SwinUNet(
        img_size=[384, 832],
        in_channels=3,
        out_channels=2,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4
    )
    
    # Create test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 384, 832)
    
    print(f"Model created successfully!")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Expected output shape: ({batch_size}, 2, 384, 832)")
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.time()
    
    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Test passed: {output.shape == (batch_size, 2, 384, 832)}")
    
    return output.shape == (batch_size, 2, 384, 832)

def test_384x832_swin_mask_unet():
    """Test SwinMaskUNet with 384x832 input"""
    print("\n" + "=" * 60)
    print("Testing SwinMaskUNet with 384x832 input size")
    print("=" * 60)
    
    # Create SwinMaskUNet with 384x832 input
    model = SwinMaskUNet(
        img_size=[384, 832],
        in_channels=3,
        out_channels=20,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4
    )
    
    # Create test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 384, 832)
    
    print(f"Model created successfully!")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Expected seg shape: ({batch_size}, 20, 384, 832)")
    print(f"Expected mask shape: ({batch_size}, 1, 384, 832)")
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        seg_output, mask = model(input_tensor)
    end_time = time.time()
    
    print(f"Segmentation output shape: {seg_output.shape}")
    print(f"Mask output shape: {mask.shape}")
    print(f"Forward pass time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Test passed: {seg_output.shape == (batch_size, 20, 384, 832) and mask.shape == (batch_size, 1, 384, 832)}")
    
    return seg_output.shape == (batch_size, 20, 384, 832) and mask.shape == (batch_size, 1, 384, 832)

def test_config_384x832():
    """Test loading 384x832 Swin-UNet from config file"""
    print("\n" + "=" * 60)
    print("Testing 384x832 Swin-UNet from config file")
    print("=" * 60)
    
    try:
        # Load config
        config = init_config("configs/sintel_swin_unet.json")
        
        # Get model from config
        model = get_model(config.model)
        
        # Create test input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 384, 832)
        
        print(f"Config loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Expected output shape: ({batch_size}, 2, 384, 832)")
        
        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        end_time = time.time()
        
        print(f"Output shape: {output.shape}")
        print(f"Forward pass time: {(end_time - start_time)*1000:.2f} ms")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Test passed: {output.shape == (batch_size, 2, 384, 832)}")
        
        return output.shape == (batch_size, 2, 384, 832)
        
    except Exception as e:
        print(f"Error loading config: {e}")
        return False

def test_memory_usage():
    """Test memory usage with 384x832 input"""
    print("\n" + "=" * 60)
    print("Testing memory usage with 384x832 input")
    print("=" * 60)
    
    # Create model
    model = SwinUNet(
        img_size=[384, 832],
        in_channels=3,
        out_channels=2,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 384, 832).to(device)
    
    print(f"Device: {device}")
    print(f"Input shape: {input_tensor.shape}")
    
    # Measure memory before forward pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.time()
    
    # Measure memory after forward pass
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_used = memory_after - memory_before
        print(f"GPU memory used: {memory_used:.2f} MB")
    
    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Test completed successfully!")
    return True

def measure_inference_speed(model, input_tensor):
    # warmup
    with torch.no_grad():
        _ = model(input_tensor)
    # 计时第二次推理
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        _ = model(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    print(f"Second inference time: {(end - start) * 1000:.2f} ms")

def measure_train_speed(model, input_tensor, target):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    # warmup
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    # 计时第二次训练
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    print(f"Second train (forward+backward+step) time: {(end - start) * 1000:.2f} ms")

def main():
    """Run all tests for 384x832 Swin-UNet"""
    print("🚀 Swin-UNet 384x832 Implementation Tests")
    print("Testing the Swin-UNet model with 384x832 input size")
    
    tests = [
        test_384x832_swin_unet,
        test_384x832_swin_mask_unet,
        test_config_384x832,
        test_memory_usage
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Swin-UNet 384x832 implementation is working correctly.")
        print("\nTo use the 384x832 Swin-UNet:")
        print("1. Use the config file: python3 train.py -c configs/sintel_swin_unet.json")
        print("2. The model will automatically use 384x832 input size")
        print("3. Make sure your data is properly resized to 384x832")
    else:
        print("❌ Some tests failed. Please check the implementation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNet(img_size=[384, 832], in_channels=3, out_channels=10).to(device)
    model.eval()
    input_tensor = torch.randn(4, 3, 384, 832).to(device)
    print("== Inference ==")
    measure_inference_speed(model, input_tensor)
    print("== Training ==")
    model.train()
    target = torch.randn(4, 10, 384, 832).to(device)
    measure_train_speed(model, input_tensor, target)

if __name__ == "__main__":
    main() 