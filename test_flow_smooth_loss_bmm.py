#!/usr/bin/env python3
"""
Test script for FlowSmoothLoss BMM optimization
测试FlowSmoothLoss BMM优化的脚本
"""

import torch
import torch.nn as nn
import time
import numpy as np
from losses.FlowSmoothLoss import FlowSmoothLoss
import os
import sys

def create_test_data(batch_size, height, width, device):
    """创建测试数据"""
    # 创建flow数据
    flows = []
    for b in range(batch_size):
        # 创建平滑的flow场
        x = torch.linspace(-1, 1, width, device=device)
        y = torch.linspace(-1, 1, height, device=device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        # 创建简单的线性flow
        flow_x = 0.1 * X + 0.05 * Y
        flow_y = -0.05 * X + 0.1 * Y
        
        flow = torch.stack([flow_x, flow_y], dim=0)  # (2, H, W)
        flows.append(flow)
    
    flows_tensor = torch.stack(flows, dim=0)  # (B, 2, H, W)
    
    # 创建mask数据
    masks = []
    for b in range(batch_size):
        # 创建简单的分割mask
        mask1 = torch.zeros(height, width, device=device)
        mask2 = torch.zeros(height, width, device=device)
        
        # 第一个mask覆盖左半部分
        mask1[:, :width//2] = 1.0
        # 第二个mask覆盖右半部分
        mask2[:, width//2:] = 1.0
        
        # 添加一些噪声使其更真实
        mask1 += torch.randn_like(mask1) * 0.1
        mask2 += torch.randn_like(mask2) * 0.1
        
        # 确保mask是概率分布
        mask = torch.stack([mask1, mask2], dim=0)
        mask = torch.softmax(mask, dim=0)
        masks.append(mask)
    
    masks_tensor = torch.stack(masks, dim=0)  # (B, 2, H, W)
    
    # 创建图像数据
    img_tensor = torch.randn(batch_size, 3, height, width, device=device)
    
    return flows_tensor, masks_tensor, img_tensor

def test_bmm_optimization():
    """测试BMM优化的性能提升"""
    print(f"\n{'='*60}")
    print(f"🚀 FlowSmoothLoss BMM优化性能测试 / FlowSmoothLoss BMM Optimization Performance Test")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备 / Using device: {device}")
    
    # 测试不同配置
    test_configs = [
        {"batch_size": 1, "height": 192, "width": 416, "name": "Small"},
        {"batch_size": 2, "height": 384, "width": 832, "name": "Medium"},
        {"batch_size": 4, "height": 384, "width": 832, "name": "Large"},
    ]
    
    flow_smooth_loss = FlowSmoothLoss(device)
    
    for config in test_configs:
        print(f"\n📊 测试配置 / Test config: {config['name']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Resolution: {config['height']}x{config['width']}")
        
        # 创建测试数据
        flows, masks, imgs = create_test_data(
            config['batch_size'], 
            config['height'], 
            config['width'], 
            device
        )
        
        # 预热
        print(f"  🔥 预热中 / Warming up...")
        for _ in range(3):
            try:
                flows_list = [flows]
                flows_list[0].requires_grad_(True)
                with torch.enable_grad():
                    loss = flow_smooth_loss(flows_list, imgs, imgs, masks)
                    if isinstance(loss, torch.Tensor):
                        loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"    ⚠️ 预热失败 / Warmup failed: {e}")
                break
        
        # 性能测试
        print(f"  ⏱️ 性能测试中 / Performance testing...")
        num_runs = 20
        forward_times = []
        backward_times = []
        total_times = []
        successful_runs = 0
        
        for i in range(num_runs):
            try:
                # 重新创建数据以确保梯度计算正确
                flows_list = [flows.clone()]
                flows_list[0].requires_grad_(True)
                
                # 前向传播
                forward_start = time.time()
                with torch.enable_grad():
                    loss = flow_smooth_loss(flows_list, imgs, imgs, masks)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_end = time.time()
                
                # 反向传播
                backward_start = time.time()
                if isinstance(loss, torch.Tensor):
                    loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                backward_end = time.time()
                
                end_time = time.time()
                
                # 记录时间
                forward_time = (forward_end - forward_start) * 1000  # ms
                backward_time = (backward_end - backward_start) * 1000  # ms
                total_time = (end_time - forward_start) * 1000  # ms
                
                forward_times.append(forward_time)
                backward_times.append(backward_time)
                total_times.append(total_time)
                successful_runs += 1
                
                if (i + 1) % 5 == 0:
                    print(f"    完成 {i+1}/{num_runs} 次测试 / Completed {i+1}/{num_runs} tests")
                    
            except Exception as e:
                print(f"    ⚠️ 第 {i+1} 次测试失败 / Test {i+1} failed: {e}")
                continue
        
        if successful_runs == 0:
            print(f"    ❌ 所有测试都失败了 / All tests failed")
            continue
        
        # 计算统计信息
        forward_mean = np.mean(forward_times)
        forward_std = np.std(forward_times)
        backward_mean = np.mean(backward_times)
        backward_std = np.std(backward_times)
        total_mean = np.mean(total_times)
        total_std = np.std(total_times)
        
        print(f"  📈 性能结果 / Performance results:")
        print(f"    前向传播 / Forward pass: {forward_mean:.2f} ± {forward_std:.2f} ms")
        print(f"    反向传播 / Backward pass: {backward_mean:.2f} ± {backward_std:.2f} ms")
        print(f"    总时间 / Total time: {total_mean:.2f} ± {total_std:.2f} ms")
        print(f"    成功率 / Success rate: {successful_runs}/{num_runs} ({100*successful_runs/num_runs:.1f}%)")

def test_memory_usage():
    """测试内存使用情况"""
    print(f"\n{'='*60}")
    print(f"💾 内存使用测试 / Memory Usage Test")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_smooth_loss = FlowSmoothLoss(device)
    
    if torch.cuda.is_available():
        # 测试GPU内存使用
        batch_size, height, width = 2, 384, 832
        flows, masks, imgs = create_test_data(batch_size, height, width, device)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 记录初始内存
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # 运行前向传播
        flows_list = [flows]
        flows_list[0].requires_grad_(True)
        with torch.enable_grad():
            loss = flow_smooth_loss(flows_list, imgs, imgs, masks)
        
        # 记录峰值内存
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        print(f"GPU内存使用情况 / GPU memory usage:")
        print(f"  初始内存 / Initial memory: {initial_memory:.2f} MB")
        print(f"  峰值内存 / Peak memory: {peak_memory:.2f} MB")
        print(f"  当前内存 / Current memory: {current_memory:.2f} MB")
        print(f"  内存增长 / Memory increase: {peak_memory - initial_memory:.2f} MB")
    else:
        print(f"CPU模式下无法测试GPU内存使用 / Cannot test GPU memory usage in CPU mode")

def test_numerical_stability():
    """测试数值稳定性"""
    print(f"\n{'='*60}")
    print(f"🔬 数值稳定性测试 / Numerical Stability Test")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_smooth_loss = FlowSmoothLoss(device)
    
    # 测试不同大小的数据
    test_sizes = [(96, 208), (192, 416), (384, 832)]
    
    for height, width in test_sizes:
        print(f"\n📏 测试尺寸 / Test size: {height}x{width}")
        
        # 创建测试数据
        flows, masks, imgs = create_test_data(1, height, width, device)
        
        # 测试多次运行的一致性
        losses = []
        for i in range(10):
            try:
                flows_list = [flows.clone()]
                flows_list[0].requires_grad_(True)
                with torch.enable_grad():
                    loss = flow_smooth_loss(flows_list, imgs, imgs, masks)
                if isinstance(loss, torch.Tensor):
                    losses.append(loss.item())
                else:
                    losses.append(float(loss))
            except Exception as e:
                print(f"  ⚠️ 第 {i+1} 次运行失败 / Run {i+1} failed: {e}")
                continue
        
        if len(losses) > 0:
            loss_mean = np.mean(losses)
            loss_std = np.std(losses)
            loss_cv = loss_std / loss_mean if loss_mean != 0 else float('inf')
            
            print(f"  损失均值 / Loss mean: {loss_mean:.6f}")
            print(f"  损失标准差 / Loss std: {loss_std:.6f}")
            print(f"  变异系数 / Coefficient of variation: {loss_cv:.6f}")
            
            if loss_cv < 0.1:
                print(f"  ✅ 数值稳定 / Numerically stable")
            else:
                print(f"  ⚠️ 数值不稳定 / Numerically unstable")

def main():
    """主函数"""
    print(f"🚀 FlowSmoothLoss BMM优化测试开始 / FlowSmoothLoss BMM Optimization Test Started")
    
    # 测试BMM优化性能
    test_bmm_optimization()
    
    # 测试内存使用
    test_memory_usage()
    
    # 测试数值稳定性
    test_numerical_stability()
    
    print(f"\n{'='*60}")
    print(f"✅ 所有测试完成 / All tests completed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 