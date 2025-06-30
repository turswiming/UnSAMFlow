#!/usr/bin/env python3
"""
PyTorch计算图资源分析示例 / PyTorch Computational Graph Resource Analysis Example

这个脚本展示了如何使用profiling和内存监控功能来分析计算图中占用资源最大的部分。
This script demonstrates how to use profiling and memory monitoring to analyze 
the most resource-intensive parts of the computational graph.
"""

import torch
import torch.nn as nn
import torch.profiler
from torch.utils.tensorboard import SummaryWriter
import os

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    Ensure directory exists, create if it doesn't
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def simple_profiling_example():
    """
    简单的profiling示例 / Simple profiling example
    """
    print("=== 简单Profiling示例 / Simple Profiling Example ===")
    
    # 确保输出目录存在 / Ensure output directories exist
    ensure_dir('./profiler_logs')
    ensure_dir('./trace_files')
    
    # 创建一个简单的模型 / Create a simple model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 3, 3, padding=1)
    ).cuda()
    
    # 创建输入数据 / Create input data
    x = torch.randn(1, 3, 224, 224).cuda()
    
    try:
        # 使用torch.profiler进行profiling / Use torch.profiler for profiling
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=5,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            
            for i in range(10):
                # 前向传播 / Forward pass
                output = model(x)
                
                # 反向传播 / Backward pass
                loss = output.sum()
                loss.backward()
                
                prof.step()
        
        # 打印结果 / Print results
        print("\n=== Profiling结果 / Profiling Results ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # 保存跟踪文件 / Save trace file
        trace_path = os.path.join('./trace_files', 'simple_trace.json')
        prof.export_chrome_trace(trace_path)
        print(f"\n跟踪文件已保存到: {trace_path}")
        print(f"TensorBoard日志已保存到: ./profiler_logs")
        
    except Exception as e:
        print(f"Profiling过程中出错 / Error during profiling: {str(e)}")
        raise

def memory_monitoring_example():
    """
    内存监控示例 / Memory monitoring example
    """
    print("\n=== 内存监控示例 / Memory Monitoring Example ===")
    
    if torch.cuda.is_available():
        # 清空缓存 / Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 创建一些张量 / Create some tensors
        tensors = []
        for i in range(5):
            tensor = torch.randn(1000, 1000).cuda()
            tensors.append(tensor)
            
            # 监控内存 / Monitor memory
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            print(f"张量 {i+1}: 分配={allocated:.2f}GB, 缓存={cached:.2f}GB, 最大={max_allocated:.2f}GB")
        
        # 清理 / Cleanup
        del tensors
        torch.cuda.empty_cache()
        
        final_allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"清理后: {final_allocated:.2f}GB")
    else:
        print("CUDA不可用 / CUDA not available")

def analyze_model_components():
    """
    分析模型组件的资源使用 / Analyze resource usage of model components
    """
    print("\n=== 模型组件分析 / Model Component Analysis ===")
    
    # 创建一个更复杂的模型 / Create a more complex model
    class ComplexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, 10)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = ComplexModel().cuda()
    x = torch.randn(4, 3, 224, 224).cuda()
    
    # 分别分析每个组件 / Analyze each component separately
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        
        # 前向传播 / Forward pass
        output = model(x)
        
        # 反向传播 / Backward pass
        loss = output.sum()
        loss.backward()
    
    print("\n=== 模型组件分析结果 / Model Component Analysis Results ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    print("PyTorch计算图资源分析工具 / PyTorch Computational Graph Resource Analysis Tool")
    print("=" * 70)
    
    # 运行示例 / Run examples
    simple_profiling_example()
    memory_monitoring_example()
    analyze_model_components()
    
    print("\n=== 使用说明 / Usage Instructions ===")
    print("1. 查看TensorBoard日志: tensorboard --logdir=./profiler_logs")
    print("2. 在Chrome中打开trace.json文件查看详细时间线")
    print("3. 使用nvtop或nvidia-smi监控GPU使用情况")
    print("4. 在训练配置中设置 enable_profiling=True 和 monitor_memory=True") 