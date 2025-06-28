#!/usr/bin/env python3
"""
Demo script for Swin-UNet implementation
Shows how to use the Swin-UNet model in the UnSAMFlow project
"""

import torch
import torch.nn as nn
from models.swin_unet import SwinUNet, SwinMaskUNet
from models.get_model import get_model, get_mask_model
from utils.config_parser import init_config
import time

def demo_basic_usage():
    """Demonstrate basic Swin-UNet usage"""
    print("=" * 60)
    print("Swin-UNet Basic Usage Demo")
    print("=" * 60)
    
    # Create a Swin-UNet model
    model = SwinUNet(
        img_size=224,
        in_channels=3,
        out_channels=2,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    
    # Create sample input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Model created successfully!")
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.time()
    
    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, output

def demo_mask_usage():
    """Demonstrate SwinMaskUNet usage"""
    print("\n" + "=" * 60)
    print("SwinMaskUNet Usage Demo")
    print("=" * 60)
    
    # Create a SwinMaskUNet model
    model = SwinMaskUNet(
        img_size=224,
        in_channels=3,
        out_channels=20,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    
    # Create sample input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Model created successfully!")
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        seg_output, mask = model(input_tensor)
    end_time = time.time()
    
    print(f"Segmentation output shape: {seg_output.shape}")
    print(f"Mask output shape: {mask.shape}")
    print(f"Forward pass time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, seg_output, mask

def demo_config_usage():
    """Demonstrate using Swin-UNet with config file"""
    print("\n" + "=" * 60)
    print("Swin-UNet Config Usage Demo")
    print("=" * 60)
    
    try:
        # Load config
        config = init_config("configs/sintel_swin_unet.json")
        
        # Get model from config
        model = get_model(config.model)
        
        # Create sample input
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        print(f"Config loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Input shape: {input_tensor.shape}")
        
        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        end_time = time.time()
        
        print(f"Output shape: {output.shape}")
        print(f"Forward pass time: {(end_time - start_time)*1000:.2f} ms")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, output
        
    except Exception as e:
        print(f"Error loading config: {e}")
        return None, None

def demo_training_setup():
    """Demonstrate training setup for Swin-UNet"""
    print("\n" + "=" * 60)
    print("Swin-UNet Training Setup Demo")
    print("=" * 60)
    
    # Create model
    model = SwinUNet(
        img_size=224,
        in_channels=3,
        out_channels=2,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    
    # Setup for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Create sample data
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
    target = torch.randn(batch_size, 2, 224, 224).to(device)
    
    print(f"Model moved to device: {device}")
    print(f"Model in training mode: {model.training}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Target shape: {target.shape}")
    
    # Training step
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Training step completed!")
    print(f"Loss: {loss.item():.6f}")
    print(f"Output shape: {output.shape}")
    
    return model, optimizer, criterion

def main():
    """Run all demos"""
    print("🚀 Swin-UNet Implementation Demo")
    print("This demo shows how to use the Swin-UNet model in UnSAMFlow")
    
    # Run demos
    demo_basic_usage()
    demo_mask_usage()
    demo_config_usage()
    demo_training_setup()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nTo use Swin-UNet in your training:")
    print("1. Use config file: python3 train.py -c configs/sintel_swin_unet.json")
    print("2. Or modify existing config to set model.type = 'swin_unet'")
    print("3. Adjust hyperparameters as needed for your specific task")

if __name__ == "__main__":
    main() 