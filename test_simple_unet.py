#!/usr/bin/env python3
"""
Test script for SimpleUNet model
"""

import torch
import torch.nn as nn
from models.simple_unet import SimpleUNet, SimpleUNetMask


def test_simple_unet():
    """Test SimpleUNet model"""
    print("Testing SimpleUNet...")
    
    # Create model
    model = SimpleUNet(
        in_channels=3,
        out_channels=2,  # For optical flow (u, v)
        features=[64, 128, 256, 512],
        bilinear=False
    )
    
    # Create dummy input
    batch_size = 2
    channels = 3
    height = 256
    width = 512
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 2, {height}, {width})")
    
    # Check if output shape is correct
    expected_shape = (batch_size, 2, height, width)
    assert output.shape == expected_shape, f"Output shape {output.shape} != expected {expected_shape}"
    
    print("✓ SimpleUNet test passed!")


def test_simple_unet_mask():
    """Test SimpleUNetMask model"""
    print("\nTesting SimpleUNetMask...")
    
    # Create model
    model = SimpleUNetMask(
        in_channels=3,
        out_channels=20,  # For mask segmentation
        features=[64, 128, 256, 512],
        bilinear=False
    )
    
    # Create dummy input
    batch_size = 2
    channels = 3
    height = 256
    width = 512
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 20, {height}, {width})")
    
    # Check if output shape is correct
    expected_shape = (batch_size, 20, height, width)
    assert output.shape == expected_shape, f"Output shape {output.shape} != expected {expected_shape}"
    
    print("✓ SimpleUNetMask test passed!")


def test_model_parameters():
    """Test model parameters count"""
    print("\nTesting model parameters...")
    
    # SimpleUNet
    model1 = SimpleUNet(
        in_channels=3,
        out_channels=2,
        features=[64, 128, 256, 512],
        bilinear=False
    )
    
    # SimpleUNetMask
    model2 = SimpleUNetMask(
        in_channels=3,
        out_channels=20,
        features=[64, 128, 256, 512],
        bilinear=False
    )
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    
    print(f"SimpleUNet parameters: {params1:,}")
    print(f"SimpleUNetMask parameters: {params2:,}")
    
    print("✓ Parameter count test passed!")


def test_different_sizes():
    """Test model with different input sizes"""
    print("\nTesting different input sizes...")
    
    model = SimpleUNet(
        in_channels=3,
        out_channels=2,
        features=[64, 128, 256, 512],
        bilinear=False
    )
    
    test_sizes = [(128, 128), (256, 256), (512, 512), (256, 512), (512, 256)]
    
    for height, width in test_sizes:
        x = torch.randn(1, 3, height, width)
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (1, 2, height, width)
        assert output.shape == expected_shape, f"Size {height}x{width}: Output shape {output.shape} != expected {expected_shape}"
        print(f"✓ Size {height}x{width} passed!")
    
    print("✓ Different sizes test passed!")


if __name__ == "__main__":
    print("=" * 50)
    print("SimpleUNet Model Tests")
    print("=" * 50)
    
    try:
        test_simple_unet()
        test_simple_unet_mask()
        test_model_parameters()
        test_different_sizes()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise 