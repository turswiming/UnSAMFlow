import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
import os


def log_mask_histogram(mask_output, summary_writer, step, tag_prefix="mask_histogram", 
                      num_bins=50, save_images=True, save_dir=None):
    """
    记录mask输出的直方图到TensorBoard
    
    Args:
        mask_output: torch.Tensor, shape (B, C, H, W) - mask模型输出
        summary_writer: SummaryWriter - TensorBoard写入器
        step: int - 当前训练步数
        tag_prefix: str - 标签前缀
        num_bins: int - 直方图bin数量
        save_images: bool - 是否保存直方图图片
        save_dir: str - 保存图片的目录
    """
    
    if not isinstance(mask_output, torch.Tensor):
        return
    
    # 确保在CPU上处理
    mask_np = mask_output.detach().cpu().numpy()
    
    # 获取维度信息
    batch_size, num_channels, height, width = mask_np.shape
    
    # 记录每个通道的直方图
    for channel in range(min(num_channels, 20)):  # 限制最多20个通道
        channel_data = mask_np[:, channel, :, :].flatten()
        
        # 计算直方图
        hist, bin_edges = np.histogram(channel_data, bins=num_bins, range=(channel_data.min(), channel_data.max()))
        
        # 记录到TensorBoard
        tag = f"{tag_prefix}/channel_{channel:02d}"
        summary_writer.add_histogram(tag, channel_data, step)
        
        # 记录统计信息
        summary_writer.add_scalar(f"{tag_prefix}/channel_{channel:02d}_mean", channel_data.mean(), step)
        summary_writer.add_scalar(f"{tag_prefix}/channel_{channel:02d}_std", channel_data.std(), step)
        summary_writer.add_scalar(f"{tag_prefix}/channel_{channel:02d}_min", channel_data.min(), step)
        summary_writer.add_scalar(f"{tag_prefix}/channel_{channel:02d}_max", channel_data.max(), step)
    
    # 记录整体统计信息
    all_data = mask_np.flatten()
    summary_writer.add_scalar(f"{tag_prefix}/overall_mean", all_data.mean(), step)
    summary_writer.add_scalar(f"{tag_prefix}/overall_std", all_data.std(), step)
    summary_writer.add_scalar(f"{tag_prefix}/overall_min", all_data.min(), step)
    summary_writer.add_scalar(f"{tag_prefix}/overall_max", all_data.max(), step)
    
    # 记录每个batch的统计信息
    for batch in range(min(batch_size, 4)):  # 限制最多4个batch
        batch_data = mask_np[batch].flatten()
        summary_writer.add_scalar(f"{tag_prefix}/batch_{batch}_mean", batch_data.mean(), step)
        summary_writer.add_scalar(f"{tag_prefix}/batch_{batch}_std", batch_data.std(), step)
    
    # 保存直方图图片
    if save_images and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Mask Output Histograms - Step {step}', fontsize=16)
        
        # 整体直方图
        axes[0, 0].hist(all_data, bins=num_bins, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Overall Distribution')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 前几个通道的直方图
        for i in range(min(3, num_channels)):
            channel_data = mask_np[:, i, :, :].flatten()
            axes[0, 1].hist(channel_data, bins=num_bins, alpha=0.7, 
                           label=f'Channel {i}', edgecolor='black')
        axes[0, 1].set_title('First 3 Channels')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 统计信息表格
        stats_text = f"""
        Overall Statistics:
        Mean: {all_data.mean():.4f}
        Std:  {all_data.std():.4f}
        Min:  {all_data.min():.4f}
        Max:  {all_data.max():.4f}
        
        Shape: {mask_np.shape}
        """
        axes[1, 0].text(0.1, 0.5, stats_text, transform=axes[1, 0].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 0].set_title('Statistics')
        axes[1, 0].axis('off')
        
        # 通道均值分布
        channel_means = [mask_np[:, i, :, :].mean() for i in range(min(10, num_channels))]
        axes[1, 1].bar(range(len(channel_means)), channel_means, alpha=0.7, color='green')
        axes[1, 1].set_title('Channel Means (First 10)')
        axes[1, 1].set_xlabel('Channel')
        axes[1, 1].set_ylabel('Mean Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(save_dir, f'mask_histogram_step_{step:06d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 记录图片到TensorBoard
        summary_writer.add_figure(f"{tag_prefix}/histogram_plot", fig, step)


def log_mask_softmax_histogram(mask_logits, summary_writer, step, tag_prefix="mask_softmax_histogram"):
    """
    记录softmax后的mask输出直方图
    
    Args:
        mask_logits: torch.Tensor, shape (B, C, H, W) - mask模型原始输出
        summary_writer: SummaryWriter - TensorBoard写入器
        step: int - 当前训练步数
        tag_prefix: str - 标签前缀
    """
    
    if not isinstance(mask_logits, torch.Tensor):
        return
    
    # 计算softmax
    softmax_mask = torch.softmax(mask_logits, dim=1)
    mask_np = softmax_mask.detach().cpu().numpy()
    
    # 获取维度信息
    batch_size, num_channels, height, width = mask_np.shape
    
    # 记录每个通道的直方图
    for channel in range(min(num_channels, 20)):
        channel_data = mask_np[:, channel, :, :].flatten()
        
        # 记录到TensorBoard
        tag = f"{tag_prefix}/channel_{channel:02d}"
        summary_writer.add_histogram(tag, channel_data, step)
        
        # 记录统计信息
        summary_writer.add_scalar(f"{tag_prefix}/channel_{channel:02d}_mean", channel_data.mean(), step)
        summary_writer.add_scalar(f"{tag_prefix}/channel_{channel:02d}_std", channel_data.std(), step)
        summary_writer.add_scalar(f"{tag_prefix}/channel_{channel:02d}_min", channel_data.min(), step)
        summary_writer.add_scalar(f"{tag_prefix}/channel_{channel:02d}_max", channel_data.max(), step)
    
    # 记录整体统计信息
    all_data = mask_np.flatten()
    summary_writer.add_scalar(f"{tag_prefix}/overall_mean", all_data.mean(), step)
    summary_writer.add_scalar(f"{tag_prefix}/overall_std", all_data.std(), step)
    summary_writer.add_scalar(f"{tag_prefix}/overall_min", all_data.min(), step)
    summary_writer.add_scalar(f"{tag_prefix}/overall_max", all_data.max(), step)
    
    # 记录最大概率值分布
    max_probs = mask_np.max(axis=1).flatten()  # 每个像素的最大概率
    summary_writer.add_histogram(f"{tag_prefix}/max_probability", max_probs, step)
    summary_writer.add_scalar(f"{tag_prefix}/max_prob_mean", max_probs.mean(), step)
    summary_writer.add_scalar(f"{tag_prefix}/max_prob_std", max_probs.std(), step)


def log_mask_argmax_distribution(mask_logits, summary_writer, step, tag_prefix="mask_argmax_distribution"):
    """
    记录argmax后的mask类别分布
    
    Args:
        mask_logits: torch.Tensor, shape (B, C, H, W) - mask模型原始输出
        summary_writer: SummaryWriter - TensorBoard写入器
        step: int - 当前训练步数
        tag_prefix: str - 标签前缀
    """
    
    if not isinstance(mask_logits, torch.Tensor):
        return
    
    # 计算argmax
    argmax_mask = torch.argmax(mask_logits, dim=1)
    mask_np = argmax_mask.detach().cpu().numpy()
    
    # 获取维度信息
    batch_size, height, width = mask_np.shape
    num_classes = mask_logits.shape[1]
    
    # 记录每个batch的类别分布
    for batch in range(min(batch_size, 4)):
        batch_data = mask_np[batch].flatten()
        
        # 计算类别计数
        class_counts = np.bincount(batch_data, minlength=num_classes)
        
        # 记录每个类别的像素数量
        for class_id in range(min(num_classes, 20)):
            summary_writer.add_scalar(
                f"{tag_prefix}/batch_{batch}_class_{class_id}_pixels", 
                class_counts[class_id], step
            )
        
        # 记录类别分布直方图
        summary_writer.add_histogram(f"{tag_prefix}/batch_{batch}_class_distribution", 
                                   batch_data, step)
    
    # 记录整体类别分布
    all_data = mask_np.flatten()
    class_counts = np.bincount(all_data, minlength=num_classes)
    
    for class_id in range(min(num_classes, 20)):
        summary_writer.add_scalar(
            f"{tag_prefix}/overall_class_{class_id}_pixels", 
            class_counts[class_id], step
        )
    
    summary_writer.add_histogram(f"{tag_prefix}/overall_class_distribution", all_data, step) 