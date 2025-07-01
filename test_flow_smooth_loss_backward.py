#!/usr/bin/env python3
"""
Test script for FlowSmoothLoss backward propagation timing
专门测试FlowSmoothLoss反向传播时间的脚本
"""

import torch
import torch.nn as nn
import time
import numpy as np
from losses.FlowSmoothLoss import FlowSmoothLoss
from models.swin_unet import SwinUNet
from models.get_model import get_mask_model
from utils.config_parser import init_config

def create_stable_flow_data(batch_size, height, width, device):
    """创建稳定的flow数据，避免数值问题"""
    # 创建更稳定的flow数据
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
    
    return torch.stack(flows, dim=0)  # (B, 2, H, W)

def create_stable_mask_data(batch_size, height, width, device):
    """创建稳定的mask数据"""
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
    
    return torch.stack(masks, dim=0)  # (B, 2, H, W)

def test_flow_smooth_loss_backward_timing():
    """测试FlowSmoothLoss反向传播时间"""
    print("=" * 80)
    print("🚀 FlowSmoothLoss 反向传播时间测试 / FlowSmoothLoss Backward Propagation Timing Test")
    print("=" * 80)
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备 / Using device: {device}")
    
    # 创建FlowSmoothLoss
    flow_smooth_loss = FlowSmoothLoss(device)
    print(f"✅ 成功创建FlowSmoothLoss / Successfully created FlowSmoothLoss")
    
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
            out_channels=2,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=4
        ).to(device)
        print(f"✅ 使用默认配置创建模型 / Created model with default config")
    
    # 设置模型为训练模式
    model.train()
    
    # 测试不同的batch size和输入尺寸
    test_configs = [
        {"batch_size": 1, "height": 384, "width": 832, "name": "1x384x832"},
        {"batch_size": 2, "height": 384, "width": 832, "name": "2x384x832"},
        {"batch_size": 4, "height": 384, "width": 832, "name": "4x384x832"},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"测试配置 / Test config: {config['name']}")
        print(f"{'='*60}")
        
        # 创建输入数据
        input_tensor = torch.randn(
            config['batch_size'], 3, config['height'], config['width']
        ).to(device)
        
        # 创建稳定的flow数据
        flows_12 = [create_stable_flow_data(config['batch_size'], config['height'], config['width'], device)]
        flows_12[0].requires_grad_(True)
        
        # 创建稳定的mask数据
        mask = create_stable_mask_data(config['batch_size'], config['height'], config['width'], device)
        
        print(f"输入尺寸 / Input shape: {input_tensor.shape}")
        print(f"Flow尺寸 / Flow shape: {flows_12[0].shape}")
        print(f"Mask尺寸 / Mask shape: {mask.shape}")
        print(f"Flow requires_grad: {flows_12[0].requires_grad}")
        
        # 预热
        print("🔥 预热中 / Warming up...")
        for _ in range(3):
            try:
                with torch.enable_grad():
                    loss = flow_smooth_loss(flows_12, input_tensor, input_tensor, mask)
                    loss.backward()
            except Exception as e:
                print(f"⚠️ 预热时出现错误 / Error during warmup: {e}")
                break
        
        # 同步GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 测试多次并记录时间
        num_runs = 10
        forward_times = []
        backward_times = []
        total_times = []
        
        print(f"🔄 开始测试 {num_runs} 次 / Starting {num_runs} test runs...")
        
        successful_runs = 0
        for i in range(num_runs):
            try:
                # 重新创建稳定的flow数据
                flows_12 = [create_stable_flow_data(config['batch_size'], config['height'], config['width'], device)]
                flows_12[0].requires_grad_(True)
                
                # 记录开始时间
                start_time = time.time()
                
                # 前向传播
                forward_start = time.time()
                with torch.enable_grad():
                    loss = flow_smooth_loss(flows_12, input_tensor, input_tensor, mask)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_end = time.time()
                
                # 反向传播
                backward_start = time.time()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                backward_end = time.time()
                
                end_time = time.time()
                
                # 记录时间
                forward_time = (forward_end - forward_start) * 1000  # ms
                backward_time = (backward_end - backward_start) * 1000  # ms
                total_time = (end_time - start_time) * 1000  # ms
                
                forward_times.append(forward_time)
                backward_times.append(backward_time)
                total_times.append(total_time)
                successful_runs += 1
                
                if (i + 1) % 5 == 0:
                    print(f"  完成 {i+1}/{num_runs} 次测试 / Completed {i+1}/{num_runs} tests")
                    
            except Exception as e:
                print(f"⚠️ 第 {i+1} 次测试失败 / Test {i+1} failed: {e}")
                continue
        
        if successful_runs == 0:
            print(f"❌ 所有测试都失败了 / All tests failed")
            continue
        
        # 计算统计信息
        forward_mean = np.mean(forward_times)
        forward_std = np.std(forward_times)
        backward_mean = np.mean(backward_times)
        backward_std = np.std(backward_times)
        total_mean = np.mean(total_times)
        total_std = np.std(total_times)
        
        # 计算FPS
        fps = 1000 / total_mean
        
        print(f"\n📊 测试结果 / Test Results (成功 {successful_runs}/{num_runs} 次):")
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
            'backward_ratio': (backward_mean/total_mean)*100,
            'successful_runs': successful_runs
        })
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"📈 测试总结 / Test Summary")
    print(f"{'='*80}")
    
    print(f"{'配置':<15} {'前向(ms)':<12} {'反向(ms)':<12} {'总计(ms)':<12} {'FPS':<8} {'反向占比':<10} {'成功率':<8}")
    print(f"{'Config':<15} {'Forward':<12} {'Backward':<12} {'Total':<12} {'FPS':<8} {'Bwd%':<10} {'Success':<8}")
    print("-" * 90)
    
    for result in results:
        print(f"{result['config']:<15} "
              f"{result['forward_mean']:<12.2f} "
              f"{result['backward_mean']:<12.2f} "
              f"{result['total_mean']:<12.2f} "
              f"{result['fps']:<8.2f} "
              f"{result['backward_ratio']:<10.1f}% "
              f"{result['successful_runs']:<8}")
    
    # 内存使用情况
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        print(f"\n💾 GPU内存使用 / GPU Memory Usage:")
        print(f"  已分配 / Allocated: {memory_allocated:.2f} MB")
        print(f"  已保留 / Reserved: {memory_reserved:.2f} MB")
    
    print(f"\n✅ 测试完成 / Test completed!")
    return results

def test_flow_smooth_loss_components():
    """测试FlowSmoothLoss的各个组件时间"""
    print(f"\n{'='*60}")
    print(f"🔧 FlowSmoothLoss组件时间测试 / FlowSmoothLoss Component Timing Test")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_smooth_loss = FlowSmoothLoss(device)
    
    # 测试数据
    batch_size = 2
    height, width = 384, 832
    flows_12 = [create_stable_flow_data(batch_size, height, width, device)]
    flows_12[0].requires_grad_(True)
    input_tensor = torch.randn(batch_size, 3, height, width).to(device)
    mask = create_stable_mask_data(batch_size, height, width, device)
    
    print(f"测试数据尺寸 / Test data shapes:")
    print(f"  Flow: {flows_12[0].shape}")
    print(f"  Input: {input_tensor.shape}")
    print(f"  Mask: {mask.shape}")
    
    # 测试embedding构建时间
    print(f"\n🔧 测试embedding构建时间 / Testing embedding construction time...")
    num_runs = 10
    embedding_times = []
    
    for i in range(num_runs):
        start_time = time.time()
        embedding = flow_smooth_loss.construct_embedding((height, width))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        embedding_times.append((end_time - start_time) * 1000)
    
    embedding_mean = np.mean(embedding_times)
    embedding_std = np.std(embedding_times)
    print(f"  Embedding构建时间 / Embedding construction time: {embedding_mean:.2f} ± {embedding_std:.2f} ms")
    

def test_flow_smooth_loss_memory():
    """测试FlowSmoothLoss内存使用"""
    print(f"\n{'='*60}")
    print(f"💾 FlowSmoothLoss内存测试 / FlowSmoothLoss Memory Test")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        print("❌ 需要GPU进行内存测试 / GPU required for memory test")
        return
    
    # 清空GPU缓存
    torch.cuda.empty_cache()
    
    flow_smooth_loss = FlowSmoothLoss(device)
    
    # 测试不同batch size的内存使用
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        try:
            # 清空缓存
            torch.cuda.empty_cache()
            
            # 创建数据
            flows_12 = [create_stable_flow_data(batch_size, 384, 832, device)]
            flows_12[0].requires_grad_(True)
            input_tensor = torch.randn(batch_size, 3, 384, 832).to(device)
            mask = create_stable_mask_data(batch_size, 384, 832, device)
            
            # 记录内存使用
            memory_before = torch.cuda.memory_allocated() / 1024**2
            
            # 前向传播
            with torch.enable_grad():
                loss = flow_smooth_loss(flows_12, input_tensor, input_tensor, mask)
            memory_after_forward = torch.cuda.memory_allocated() / 1024**2
            
            # 反向传播
            loss.backward()
            memory_after_backward = torch.cuda.memory_allocated() / 1024**2
            
            print(f"Batch size {batch_size}:")
            print(f"  前向传播内存 / Forward memory: {memory_after_forward - memory_before:.2f} MB")
            print(f"  反向传播内存 / Backward memory: {memory_after_backward - memory_after_forward:.2f} MB")
            print(f"  总内存 / Total memory: {memory_after_backward:.2f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Batch size {batch_size}: GPU内存不足 / Out of GPU memory")
                break
            else:
                print(f"❌ Batch size {batch_size}: {e}")

def test_flow_smooth_loss_vs_other_losses():
    """比较FlowSmoothLoss与其他损失函数的时间"""
    print(f"\n{'='*60}")
    print(f"⚖️ 损失函数时间比较 / Loss Function Timing Comparison")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建不同的损失函数
    flow_smooth_loss = FlowSmoothLoss(device)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    # 测试数据
    batch_size = 2
    flows_12 = [create_stable_flow_data(batch_size, 384, 832, device)]
    flows_12[0].requires_grad_(True)
    input_tensor = torch.randn(batch_size, 3, 384, 832).to(device)
    mask = create_stable_mask_data(batch_size, 384, 832, device)
    
    # 创建目标数据
    target = create_stable_flow_data(batch_size, 384, 832, device)
    target.requires_grad_(True)
    
    num_runs = 10
    loss_functions = [
        ("FlowSmoothLoss", flow_smooth_loss, lambda: flow_smooth_loss(flows_12, input_tensor, input_tensor, mask)),
        ("MSE Loss", mse_loss, lambda: mse_loss(flows_12[0], target)),
        ("L1 Loss", l1_loss, lambda: l1_loss(flows_12[0], target))
    ]
    
    results = []
    
    for name, loss_func, loss_compute in loss_functions:
        print(f"\n🔧 测试 {name} / Testing {name}...")
        
        # 预热
        for _ in range(3):
            try:
                with torch.enable_grad():
                    loss = loss_compute()
                    loss.backward()
            except Exception as e:
                print(f"⚠️ 预热时出现错误 / Error during warmup: {e}")
                break
        
        # 测试时间
        forward_times = []
        backward_times = []
        successful_runs = 0
        
        for i in range(num_runs):
            try:
                # 重新创建数据以确保梯度计算正确
                if name == "FlowSmoothLoss":
                    flows_12 = [create_stable_flow_data(batch_size, 384, 832, device)]
                    flows_12[0].requires_grad_(True)
                    loss_compute = lambda: flow_smooth_loss(flows_12, input_tensor, input_tensor, mask)
                else:
                    flows_12 = [create_stable_flow_data(batch_size, 384, 832, device)]
                    flows_12[0].requires_grad_(True)
                    target = create_stable_flow_data(batch_size, 384, 832, device)
                    target.requires_grad_(True)
                    if name == "MSE Loss":
                        loss_compute = lambda: mse_loss(flows_12[0], target)
                    else:
                        loss_compute = lambda: l1_loss(flows_12[0], target)
                
                # 前向传播
                forward_start = time.time()
                with torch.enable_grad():
                    loss = loss_compute()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_end = time.time()
                
                # 反向传播
                backward_start = time.time()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                backward_end = time.time()
                
                forward_times.append((forward_end - forward_start) * 1000)
                backward_times.append((backward_end - backward_start) * 1000)
                successful_runs += 1
                
            except Exception as e:
                print(f"⚠️ 第 {i+1} 次测试失败 / Test {i+1} failed: {e}")
                continue
        
        if successful_runs == 0:
            print(f"❌ 所有测试都失败了 / All tests failed")
            continue
        
        forward_mean = np.mean(forward_times)
        backward_mean = np.mean(backward_times)
        total_mean = forward_mean + backward_mean
        
        print(f"  前向传播 / Forward: {forward_mean:.2f} ms")
        print(f"  反向传播 / Backward: {backward_mean:.2f} ms")
        print(f"  总时间 / Total: {total_mean:.2f} ms")
        print(f"  成功率 / Success rate: {successful_runs}/{num_runs}")
        
        results.append({
            'name': name,
            'forward': forward_mean,
            'backward': backward_mean,
            'total': total_mean,
            'successful_runs': successful_runs
        })
    
    # 打印比较结果
    print(f"\n📊 损失函数时间比较 / Loss function timing comparison:")
    print(f"{'损失函数':<15} {'前向(ms)':<12} {'反向(ms)':<12} {'总计(ms)':<12} {'成功率':<8}")
    print(f"{'Loss Function':<15} {'Forward':<12} {'Backward':<12} {'Total':<12} {'Success':<8}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<15} "
              f"{result['forward']:<12.2f} "
              f"{result['backward']:<12.2f} "
              f"{result['total']:<12.2f} "
              f"{result['successful_runs']:<8}")

if __name__ == "__main__":
    # 运行FlowSmoothLoss反向传播时间测试
    results = test_flow_smooth_loss_backward_timing()
    
    # 运行组件时间测试
    test_flow_smooth_loss_components()
    
    # 运行内存测试
    test_flow_smooth_loss_memory()
    
    # 运行损失函数比较测试
    test_flow_smooth_loss_vs_other_losses()
    
    print(f"\n🎉 所有测试完成 / All tests completed!") 