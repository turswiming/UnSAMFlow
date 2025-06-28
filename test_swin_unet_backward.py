#!/usr/bin/env python3
"""
Test script for SwinUNet backward propagation timing
专门测试SwinUNet反向传播时间的脚本
"""

import torch
import torch.nn as nn
import time
import numpy as np
from models.swin_unet import SwinUNet
from models.get_model import get_mask_model
from utils.config_parser import init_config

def test_backward_timing():
    """测试SwinUNet反向传播时间"""
    print("=" * 80)
    print("🚀 SwinUNet 反向传播时间测试 / SwinUNet Backward Propagation Timing Test")
    print("=" * 80)
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备 / Using device: {device}")
    
    # 从配置文件加载模型
    try:
        config = init_config("configs/sintel_swin_unet.json")
        model = get_mask_model(config.mask_model)
        model = model.to(device)
        print(f"✅ 成功从配置文件加载模型 / Successfully loaded model from config")
        print(f"模型类型 / Model type: {type(model).__name__}")
    except Exception as e:
        print(f"❌ 配置文件加载失败 / Failed to load config: {e}")
        # 使用默认配置创建模型
        model = SwinUNet(
            img_size=[384, 832],
            in_channels=3,
            out_channels=20,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=4
        ).to(device)
        print(f"✅ 使用默认配置创建模型 / Created model with default config")
    
    # 设置模型为训练模式
    model.train()
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 测试不同的batch size和输入尺寸
    test_configs = [
        {"batch_size": 1, "height": 384, "width": 832, "name": "1x384x832"},
        {"batch_size": 2, "height": 384, "width": 832, "name": "2x384x832"},
        {"batch_size": 4, "height": 384, "width": 832, "name": "4x384x832"},
        {"batch_size": 8, "height": 384, "width": 832, "name": "8x384x832"},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"测试配置 / Test config: {config['name']}")
        print(f"{'='*60}")
        
        # 创建输入和目标
        input_tensor = torch.randn(
            config['batch_size'], 3, config['height'], config['width']
        ).to(device)
        
        # 根据模型输出创建目标
        with torch.no_grad():
            if hasattr(model, 'forward') and callable(getattr(model, 'forward', None)):
                output = model(input_tensor)
                if isinstance(output, tuple):
                    # SwinMaskUNet返回两个输出
                    target = torch.randn_like(output[0])
                else:
                    # SwinUNet返回单个输出
                    target = torch.randn_like(output)
            else:
                # 默认目标
                target = torch.randn(config['batch_size'], 20, config['height'], config['width']).to(device)
        
        print(f"输入尺寸 / Input shape: {input_tensor.shape}")
        print(f"目标尺寸 / Target shape: {target.shape}")
        print(f"模型参数数量 / Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 预热
        print("🔥 预热中 / Warming up...")
        for _ in range(3):
            optimizer.zero_grad()
            output = model(input_tensor)
            if isinstance(output, tuple):
                loss = criterion(output[0], target)
            else:
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # 同步GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 测试多次并记录时间
        num_runs = 10
        forward_times = []
        backward_times = []
        total_times = []
        
        print(f"🔄 开始测试 {num_runs} 次 / Starting {num_runs} test runs...")
        
        for i in range(num_runs):
            # 记录开始时间
            start_time = time.time()
            
            # 前向传播
            optimizer.zero_grad()
            forward_start = time.time()
            output = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_end = time.time()
            
            # 计算损失
            if isinstance(output, tuple):
                loss = criterion(output[0], target)
            else:
                loss = criterion(output, target)
            
            # 反向传播
            backward_start = time.time()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_end = time.time()
            
            # 优化器步骤
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            # 记录时间
            forward_time = (forward_end - forward_start) * 1000  # ms
            backward_time = (backward_end - backward_start) * 1000  # ms
            total_time = (end_time - start_time) * 1000  # ms
            
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            total_times.append(total_time)
            
            if (i + 1) % 5 == 0:
                print(f"  完成 {i+1}/{num_runs} 次测试 / Completed {i+1}/{num_runs} tests")
        
        # 计算统计信息
        forward_mean = np.mean(forward_times)
        forward_std = np.std(forward_times)
        backward_mean = np.mean(backward_times)
        backward_std = np.std(backward_times)
        total_mean = np.mean(total_times)
        total_std = np.std(total_times)
        
        # 计算FPS
        fps = 1000 / total_mean
        
        print(f"\n📊 测试结果 / Test Results:")
        print(f"  前向传播时间 / Forward time: {forward_mean:.2f} ± {forward_std:.2f} ms")
        print(f"  反向传播时间 / Backward time: {backward_mean:.2f} ± {backward_std:.2f} ms")
        print(f"  总时间 / Total time: {total_mean:.2f} ± {total_std:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        print(f"  反向传播占比 / Backward ratio: {(backward_mean/total_mean)*100:.1f}%")
        
        # 记录结果
        results.append({
            'config': config['name'],
            'forward_mean': forward_mean,
            'forward_std': forward_std,
            'backward_mean': backward_mean,
            'backward_std': backward_std,
            'total_mean': total_mean,
            'total_std': total_std,
            'fps': fps,
            'backward_ratio': (backward_mean/total_mean)*100
        })
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"📈 测试总结 / Test Summary")
    print(f"{'='*80}")
    
    print(f"{'配置':<15} {'前向(ms)':<12} {'反向(ms)':<12} {'总计(ms)':<12} {'FPS':<8} {'反向占比':<10}")
    print(f"{'Config':<15} {'Forward':<12} {'Backward':<12} {'Total':<12} {'FPS':<8} {'Bwd%':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['config']:<15} "
              f"{result['forward_mean']:<12.2f} "
              f"{result['backward_mean']:<12.2f} "
              f"{result['total_mean']:<12.2f} "
              f"{result['fps']:<8.2f} "
              f"{result['backward_ratio']:<10.1f}%")
    
    # 内存使用情况
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        print(f"\n💾 GPU内存使用 / GPU Memory Usage:")
        print(f"  已分配 / Allocated: {memory_allocated:.2f} MB")
        print(f"  已保留 / Reserved: {memory_reserved:.2f} MB")
    
    print(f"\n✅ 测试完成 / Test completed!")
    return results

def test_memory_efficiency():
    """测试内存效率"""
    print(f"\n{'='*60}")
    print(f"💾 内存效率测试 / Memory Efficiency Test")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        print("❌ 需要GPU进行内存测试 / GPU required for memory test")
        return
    
    # 清空GPU缓存
    torch.cuda.empty_cache()
    
    # 创建模型
    model = SwinUNet(
        img_size=[384, 832],
        in_channels=3,
        out_channels=20,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4
    ).to(device)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 测试不同batch size的内存使用
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        try:
            # 清空缓存
            torch.cuda.empty_cache()
            
            # 创建输入
            input_tensor = torch.randn(batch_size, 3, 384, 832).to(device)
            target = torch.randn(batch_size, 20, 384, 832).to(device)
            
            # 记录内存使用
            memory_before = torch.cuda.memory_allocated() / 1024**2
            
            # 前向传播
            output = model(input_tensor)
            memory_after_forward = torch.cuda.memory_allocated() / 1024**2
            
            # 反向传播
            loss = criterion(output, target)
            loss.backward()
            memory_after_backward = torch.cuda.memory_allocated() / 1024**2
            
            # 优化器步骤
            optimizer.step()
            memory_after_optimizer = torch.cuda.memory_allocated() / 1024**2
            
            print(f"Batch size {batch_size}:")
            print(f"  前向传播内存 / Forward memory: {memory_after_forward - memory_before:.2f} MB")
            print(f"  反向传播内存 / Backward memory: {memory_after_backward - memory_after_forward:.2f} MB")
            print(f"  优化器内存 / Optimizer memory: {memory_after_optimizer - memory_after_backward:.2f} MB")
            print(f"  总内存 / Total memory: {memory_after_optimizer:.2f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Batch size {batch_size}: GPU内存不足 / Out of GPU memory")
                break
            else:
                print(f"❌ Batch size {batch_size}: {e}")

if __name__ == "__main__":
    # 运行反向传播时间测试
    results = test_backward_timing()
    
    # 运行内存效率测试
    test_memory_efficiency()
    
    print(f"\n🎉 所有测试完成 / All tests completed!") 