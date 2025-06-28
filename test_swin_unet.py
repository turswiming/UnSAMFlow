#!/usr/bin/env python3
"""
Test script for Swin-UNet implementation
"""

import torch
import torch.nn as nn
from models.swin_unet import SwinUNet, SwinMaskUNet
from models.get_model import get_model, get_mask_model
from utils.config_parser import init_config
import json

def test_swin_unet_basic():
    """Test basic Swin-UNet functionality"""
    print("Testing basic Swin-UNet...")
    
    # Create a simple Swin-UNet
    model = SwinUNet(
        img_size=224,
        in_channels=3,
        out_channels=2,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    
    # Create test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 384, 832)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 2, 384, 832)")
    print(f"Test passed: {output.shape == (batch_size, 2, 384, 832)}")
    
    return output.shape == (batch_size, 2, 384, 224)

def test_swin_mask_unet():
    """Test SwinMaskUNet functionality"""
    print("\nTesting SwinMaskUNet...")
    
    # Create a SwinMaskUNet
    model = SwinMaskUNet(
        img_size=224,
        in_channels=3,
        out_channels=20,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    
    # Create test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 384, 832)
    
    # Forward pass
    with torch.no_grad():
        seg_output, mask = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Segmentation output shape: {seg_output.shape}")
    print(f"Mask output shape: {mask.shape}")
    print(f"Expected seg shape: ({batch_size}, 20, 384, 832)")
    print(f"Expected mask shape: ({batch_size}, 1, 384, 832)")
    print(f"Test passed: {seg_output.shape == (batch_size, 20, 384, 832) and mask.shape == (batch_size, 1, 384, 832)}")
    
    return seg_output.shape == (batch_size, 20, 384, 832) and mask.shape == (batch_size, 1, 384, 832)

def test_get_model():
    """Test get_model function with Swin-UNet"""
    print("\nTesting get_model function...")
    
    # Create a mock config
    class MockConfig:
        def __init__(self):
            self.type = "swin_unet"
            self.img_size = 224
            self.in_channels = 3
            self.out_channels = 2
            self.embed_dim = 96
            self.depths = [2, 2, 6, 2]
            self.num_heads = [3, 6, 12, 24]
            self.window_size = 7
            self.mlp_ratio = 4.0
            self.qkv_bias = True
            self.qk_scale = None
            self.drop_rate = 0.0
            self.attn_drop_rate = 0.0
            self.drop_path_rate = 0.1
            self.use_checkpoint = False
    
    config = MockConfig()
    
    # Get model
    model = get_model(config)
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 384, 832)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Model type: {type(model).__name__}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 2, 384, 832)")
    print(f"Test passed: {output.shape == (batch_size, 2, 384, 832)}")
    
    return output.shape == (batch_size, 2, 384, 832)

def test_get_mask_model():
    """Test get_mask_model function with Swin-UNet"""
    print("\nTesting get_mask_model function...")
    
    # Create a mock config
    class MockConfig:
        def __init__(self):
            self.type = "swin_unet"
            self.img_size = 224
            self.in_channels = 3
            self.out_channels = 20
            self.embed_dim = 96
            self.depths = [2, 2, 6, 2]
            self.num_heads = [3, 6, 12, 24]
            self.window_size = 7
            self.mlp_ratio = 4.0
            self.qkv_bias = True
            self.qk_scale = None
            self.drop_rate = 0.0
            self.attn_drop_rate = 0.0
            self.drop_path_rate = 0.1
            self.use_checkpoint = False
    
    config = MockConfig()
    
    # Get mask model
    model = get_mask_model(config)
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 384, 832)
    
    with torch.no_grad():
        seg_output, mask = model(input_tensor)
    
    print(f"Model type: {type(model).__name__}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Segmentation output shape: {seg_output.shape}")
    print(f"Mask output shape: {mask.shape}")
    print(f"Expected seg shape: ({batch_size}, 20, 384, 832)")
    print(f"Expected mask shape: ({batch_size}, 1, 384, 832)")
    print(f"Test passed: {seg_output.shape == (batch_size, 20, 384, 832) and mask.shape == (batch_size, 1, 384, 832)}")
    
    return seg_output.shape == (batch_size, 20, 384, 832) and mask.shape == (batch_size, 1, 384, 832)

def test_config_file():
    """Test loading Swin-UNet from config file"""
    print("\nTesting config file loading...")
    
    try:
        # Load config
        config = init_config("configs/sintel_swin_unet.json")
        
        # Get model
        model = get_model(config.model)
        
        # Test forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 384, 832)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"Config loaded successfully")
        print(f"Model type: {type(model).__name__}")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: ({batch_size}, 2, 384, 832)")
        print(f"Test passed: {output.shape == (batch_size, 2, 384, 832)}")
        
        return output.shape == (batch_size, 2, 384, 832)
        
    except Exception as e:
        print(f"Error loading config: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Swin-UNet Implementation Tests")
    print("=" * 50)
    
    tests = [
        test_swin_unet_basic,
        test_swin_mask_unet,
        test_get_model,
        test_get_mask_model,
        test_config_file
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! Swin-UNet implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 