# SimpleUNet for UnSAMFlow

## 概述 (Overview)

除了现有的SwinUNet外，我们添加了一个最简单的UNet模型用于mask分割。这个SimpleUNet基于经典的U-Net架构，具有以下特点：

- **简单高效**: 经典的编码器-解码器结构
- **灵活配置**: 可自定义特征通道数和上采样方式
- **轻量级**: 相比SwinUNet参数更少，训练更快

## 模型结构 (Model Architecture)

### SimpleUNet
- **输入**: 3通道RGB图像
- **输出**: 2通道光流图 (u, v)
- **特征**: 可配置的特征通道数 [64, 128, 256, 512]

### SimpleUNetMask  
- **输入**: 3通道RGB图像
- **输出**: 20通道mask分割图
- **特征**: 可配置的特征通道数 [64, 128, 256, 512]

## 使用方法 (Usage)

### 1. 配置文件设置 (Configuration)

在配置文件中设置模型类型为 `simple_unet`:

```json
{
    "model": {
        "type": "simple_unet",
        "in_channels": 3,
        "out_channels": 2,
        "features": [64, 128, 256, 512],
        "bilinear": false
    },
    "mask_model": {
        "type": "simple_unet", 
        "in_channels": 3,
        "out_channels": 20,
        "features": [64, 128, 256, 512],
        "bilinear": false
    }
}
```

### 2. 训练命令 (Training Commands)

使用SimpleUNet进行训练：

```bash
# Sintel数据集
python3 train.py -c configs/sintel_simple_unet.json --n_gpu=N_GPU --exp_folder=EXP_FOLDER

# KITTI数据集 (需要创建对应的配置文件)
python3 train.py -c configs/kitti_simple_unet.json --n_gpu=N_GPU --exp_folder=EXP_FOLDER
```

### 3. 测试模型 (Testing)

运行测试脚本验证模型：

```bash
cd UnSAMFlow
python3 test_simple_unet.py
```

## 配置参数 (Configuration Parameters)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `in_channels` | int | 3 | 输入通道数 |
| `out_channels` | int | 2 | 输出通道数 |
| `features` | list | [64, 128, 256, 512] | 各层特征通道数 |
| `bilinear` | bool | false | 是否使用双线性上采样 |

## 模型特点 (Model Features)

### 优点 (Advantages)
- ✅ **简单易懂**: 经典的U-Net架构，易于理解和修改
- ✅ **训练快速**: 参数较少，训练速度比SwinUNet快
- ✅ **内存友好**: 占用显存较少
- ✅ **灵活配置**: 可自定义特征通道数和上采样方式

### 适用场景 (Use Cases)
- 快速原型开发
- 资源受限的环境
- 需要轻量级模型的场景
- 对模型复杂度有要求的应用

## 与SwinUNet的对比 (Comparison with SwinUNet)

| 特性 | SimpleUNet | SwinUNet |
|------|------------|----------|
| 架构复杂度 | 简单 | 复杂 |
| 参数量 | 较少 | 较多 |
| 训练速度 | 快 | 慢 |
| 内存占用 | 低 | 高 |
| 性能 | 基础 | 优秀 |
| 适用场景 | 轻量级应用 | 高性能应用 |

## 示例配置文件 (Example Configurations)

### 轻量级配置 (Lightweight Configuration)
```json
{
    "model": {
        "type": "simple_unet",
        "features": [32, 64, 128, 256],
        "bilinear": true
    }
}
```

### 标准配置 (Standard Configuration)
```json
{
    "model": {
        "type": "simple_unet", 
        "features": [64, 128, 256, 512],
        "bilinear": false
    }
}
```

### 高性能配置 (High Performance Configuration)
```json
{
    "model": {
        "type": "simple_unet",
        "features": [128, 256, 512, 1024],
        "bilinear": false
    }
}
```

## 故障排除 (Troubleshooting)

### 常见问题 (Common Issues)

1. **内存不足**: 减少batch_size或使用更小的features配置
2. **训练速度慢**: 使用bilinear=True或减少features
3. **模型性能差**: 增加features或使用更复杂的配置

### 调试建议 (Debugging Tips)

- 使用测试脚本验证模型结构
- 检查配置文件中的参数设置
- 监控训练过程中的内存使用情况

## 贡献 (Contributions)

欢迎提交改进建议和bug报告！

---

**English Version:**

# SimpleUNet for UnSAMFlow

## Overview

In addition to the existing SwinUNet, we have added a simplest UNet model for mask segmentation. This SimpleUNet is based on the classic U-Net architecture with the following features:

- **Simple and Efficient**: Classic encoder-decoder structure
- **Flexible Configuration**: Customizable feature channels and upsampling methods
- **Lightweight**: Fewer parameters than SwinUNet, faster training

## Model Architecture

### SimpleUNet
- **Input**: 3-channel RGB image
- **Output**: 2-channel optical flow (u, v)
- **Features**: Configurable feature channels [64, 128, 256, 512]

### SimpleUNetMask
- **Input**: 3-channel RGB image  
- **Output**: 20-channel mask segmentation
- **Features**: Configurable feature channels [64, 128, 256, 512]

## Usage

### 1. Configuration Setup

Set the model type to `simple_unet` in the configuration file:

```json
{
    "model": {
        "type": "simple_unet",
        "in_channels": 3,
        "out_channels": 2,
        "features": [64, 128, 256, 512],
        "bilinear": false
    }
}
```

### 2. Training Commands

Train with SimpleUNet:

```bash
# Sintel dataset
python3 train.py -c configs/sintel_simple_unet.json --n_gpu=N_GPU --exp_folder=EXP_FOLDER
```

### 3. Testing

Run the test script to verify the model:

```bash
cd UnSAMFlow
python3 test_simple_unet.py
```

## Features

### Advantages
- ✅ **Simple and Understandable**: Classic U-Net architecture, easy to understand and modify
- ✅ **Fast Training**: Fewer parameters, faster training than SwinUNet
- ✅ **Memory Friendly**: Lower GPU memory usage
- ✅ **Flexible Configuration**: Customizable feature channels and upsampling

### Use Cases
- Rapid prototyping
- Resource-constrained environments
- Applications requiring lightweight models
- Scenarios with model complexity requirements 