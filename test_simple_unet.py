#!/usr/bin/env python3
"""
Test script for SimpleUNet model
"""
import time
import torch
import torch.nn as nn
from models.simple_unet import SimpleUNet, SimpleUNetMask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 2
channels = 3
height = 448
width = 1024
def test_simple_unet():
    """Test SimpleUNet model"""
    print("Testing SimpleUNet...")
    
    # Create model
    model = SimpleUNet(
        in_channels=3,
        out_channels=20,  # For optical flow (u, v)
        features=[64, 128, 256, 512],
        bilinear=False
    )
    model.to(device)
    # Create dummy input


    x = torch.randn(batch_size, channels, height, width).to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 20, {height}, {width})")
    
    # Check if output shape is correct
    expected_shape = (batch_size, 20, height, width)
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
        bilinear=True
    )
    model.to(device)
    
    # Create dummy input

    x = torch.randn(batch_size, channels, height, width).to(device)
    
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
    model1.to(device)
    # SimpleUNetMask
    model2 = SimpleUNetMask(
        in_channels=3,
        out_channels=20,
        features=[64, 128, 256, 512],
        bilinear=True
    )
    model2.to(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    
    print(f"SimpleUNet parameters: {params1:,}")
    print(f"SimpleUNetMask parameters: {params2:,}")
    
    print("✓ Parameter count test passed!")




def test_inference_speed():
    """Test inference speed"""
    print("\nTesting inference speed...")
    
    model = SimpleUNet(
        in_channels=3,
        out_channels=2,
        features=[64, 128, 256, 512],
        bilinear=False
    )
    model.to(device)
    
    # Create dummy input
    x = torch.randn(batch_size, channels, height, width).to(device)
    
    print(f"Input shape: {x.shape}")
    infer_iter = 10
        # Forward pass  
    with torch.no_grad():
        start_time = time.time()
        for _ in range(infer_iter):
            output = model(x)
        end_time = time.time()
        inference_time = (end_time - start_time) / infer_iter
    
    print(f"Inference time: {inference_time:.4f} seconds")
    print("✓ Inference speed test passed!")


def test_backward_speed():
    """Test backward speed"""
    print("\nTesting backward speed...")
    
    model = SimpleUNet(
        in_channels=3,
        out_channels=2,
        features=[64, 128, 256, 512],
        bilinear=False
    )
    model.to(device)

    infer_iter = 10
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(infer_iter):
        x = torch.randn(batch_size, channels, height, width).to(device)
        output = model(x)
        loss = output.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    backward_time = (end_time - start_time) / infer_iter
    print(f"Backward time: {backward_time:.4f} seconds")
    print("✓ Backward speed test passed!")

if __name__ == "__main__":
    print("=" * 50)
    print("SimpleUNet Model Tests")
    print("=" * 50)
    
    try:
        test_simple_unet()
        test_simple_unet_mask()
        test_model_parameters()
        test_inference_speed()
        test_backward_speed()
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise 