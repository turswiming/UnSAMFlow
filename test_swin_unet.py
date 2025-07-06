#!/usr/bin/env python3
"""
Test script for Swin-UNet implementation
"""

import torch
import torch.nn as nn
from models.swin_unet import SwinUNet
from models.get_model import get_model, get_mask_model
from utils.config_parser import init_config
import json
import time
#print the stack trace
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
out_channels = 10
in_channels = 3
height = 448
width = 1024

def test_swin_unet_basic():
    """Test basic Swin-UNet functionality"""
    print("Testing basic Swin-UNet...")
    
    # Create a simple Swin-UNet
    model = SwinUNet(
        img_size=(height, width),
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    
    # Create test input
    input_tensor = torch.randn(batch_size, in_channels, height, width)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {out_channels}, {height}, {width})")
    print(f"Test passed: {output.shape == (batch_size, out_channels, height,width)}")
    
    return output.shape == (batch_size, out_channels, height, width)


def test_inference_speed():
    """Test inference speed"""
    print("\nTesting inference speed...")
    global height, width
    model = SwinUNet(
        img_size=(height, width),
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    model.to(device)
    
    x = torch.randn(batch_size, in_channels, height, width).to(device)
    
    print(f"Input shape: {x.shape}")
    infer_iter = 10
    with torch.no_grad():
        start_time = time.time()
        for _ in range(infer_iter):
            output = model(x)
        end_time = time.time()
        inference_time = (end_time - start_time) / infer_iter
    
    print(f"Inference time: {inference_time:.4f} seconds")
    print("✓ Inference speed test passed!")
    return True
def test_backward_speed():
    """Test backward speed"""
    print("\nTesting backward speed...")
    global height, width
    model = SwinUNet(
        img_size=(height, width),
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    model.to(device)

    # Create dummy input

    infer_iter = 10
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(infer_iter):
        x = torch.randn(batch_size, in_channels, height, width).to(device)
        output = model(x)
        loss = output.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    backward_time = (end_time - start_time) / infer_iter
    print(f"Backward time: {backward_time:.4f} seconds")
    print("✓ Backward speed test passed!")
    return True

def test_parameters():
    """Test model parameters"""
    print("\nTesting model parameters...")
    model = SwinUNet(
        img_size=(height, width),
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("Swin-UNet Implementation Tests")
    print("=" * 50)
    
    tests = [
        test_swin_unet_basic,
        test_inference_speed,
        test_backward_speed,
        test_parameters
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with error: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! Swin-UNet implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 