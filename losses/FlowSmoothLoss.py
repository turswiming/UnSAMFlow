"""
Flow Smoothness Loss implementation for scene flow prediction.

This module implements a parametric flow smoothness loss that encourages spatially
coherent flow predictions within segmented regions. It uses a quadratic flow
approximation approach to ensure smooth transitions in the predicted flow field.
"""

import torch
import functools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def normalize_global(x):
    """
    Normalize tensor globally using standard deviation.
    
    Args:
        x (torch.Tensor): Input tensor to normalize
        
    Returns:
        torch.Tensor: Normalized tensor
    """
    with torch.no_grad():
        std = x.clone().reshape(-1).std(dim=0)
        if std.max() <= 1e-6:
            std = torch.ones_like(std)
    x = x/std # (HW, 2)
    return x

def normalize_useing_other(x, points):
    """
    Normalize tensor using standard deviation from another tensor.
    
    Args:
        x (torch.Tensor): Tensor to normalize
        points (torch.Tensor): Tensor to compute normalization statistics from
        
    Returns:
        torch.Tensor: Normalized tensor
    """
    with torch.no_grad():
        std = points.clone().reshape(-1).std(dim=0)
        if std.max() <= 1e-6:
            std = torch.ones_like(std)
    return x/std

class FlowSmoothLoss():

    def __init__(self, device):
        """
        Initialize the Flow Smoothness Loss.
        
        Args:
            device (torch.device): Device to perform computations on
        """
        self.device=device
        each_mask_item_gradient = 1
        sum_mask_item_gradient = 1
        self.each_mask_item_gradient = each_mask_item_gradient/(each_mask_item_gradient+sum_mask_item_gradient)
        self.sum_mask_item_gradient = sum_mask_item_gradient/(each_mask_item_gradient+sum_mask_item_gradient)
        self.each_mask_item_loss = "L2"
        self.sum_mask_item_loss = "L2"
        if self.each_mask_item_loss in ["L1", "l1"]:
            self.each_mask_criterion = nn.L1Loss(reduction="mean").to(self.device)
        elif self.each_mask_item_loss in ["L2", "l2"]:
            self.each_mask_criterion = nn.MSELoss(reduction="mean").to(self.device)
        else:
            raise ValueError(f"Invalid loss criterion: {self.each_mask_item_loss}")
        if self.sum_mask_item_loss in ["L1", "l1"]:
            self.sum_mask_criterion = nn.L1Loss(reduction="mean").to(self.device)
        elif self.sum_mask_item_loss in ["L2", "l2"]:
            self.sum_mask_criterion = nn.MSELoss(reduction="mean").to(self.device)
        else:
            raise ValueError(f"Invalid loss criterion: {self.sum_mask_item_loss}")

        pass

    def __call__(self, pyramid_flows, img1, img2, mask):
        return self.loss(pyramid_flows, img1, img2,mask)
        
    def loss(self, pyramid_flows, img1, img2, mask):
        batch_size = len(pyramid_flows[0])
        # print(f"batch_size: {batch_size}")
        # print(f"pyramid_flows: {pyramid_flows[0].shape}")
        # print(f"img1: {img1.shape}")
        # print(f"img2: {img2.shape}")
        # print(f"mask: {mask.shape}")
        """
        batch_size: 1
        pyramid_flows: torch.Size([1, 2, 384, 832])
        img1: torch.Size([1, 3, 384, 832])
        img2: torch.Size([1, 3, 384, 832])
        mask: torch.Size([1, 2, 384, 832])
        """
        with torch.no_grad():
            uv_size = pyramid_flows[0].shape[2:]
            u = torch.arange(uv_size[0], device=self.device)
            v = torch.arange(uv_size[1], device=self.device)
            u, v = torch.meshgrid(u, v, indexing="ij")
            embedding = torch.stack([
                u, v,
                u*u, v*v,
                u*v,
                torch.ones_like(u),
            ], dim=-1).reshape(-1, 6)
        total_loss = 0.0
        for b in range(batch_size):
            # Get batch data
            pyramid_flow_b = pyramid_flows[0][b]  # (N, 2)
            mask_b = mask[b]  # (K, N)
            
            # Process mask
            
            # Normalize flow
            pyramid_flow_b = normalize_useing_other(pyramid_flow_b, pyramid_flow_b)
            pyramid_flow_b = pyramid_flow_b.permute(1,2,0)
            # Construct embedding
            
            # Initialize flow reconstruction
            flow_reconstruction = torch.zeros_like(pyramid_flow_b.reshape(-1,2))  # (N, 2)
            
            # Per-slot reconstruction
            K = mask_b.shape[0]
            reconstruction_loss = 0
            for k in range(K):
                mk = mask_b[k].unsqueeze(-1)  # (N, 1)
                mk = mk.reshape(-1,1)
                pyramid_flow_b = pyramid_flow_b.reshape(-1,2)

                Ek = embedding * mk  # Apply mask to embedding
                Fk = pyramid_flow_b * mk  # Apply mask to flow
                Ek_t = Ek.transpose(0,1)
                A = Ek_t @ Ek
                A = A + 1e-6 *torch.eye(6,device=A.device).unsqueeze(0)
                b = Ek_t @ Fk
                # Solve for parameters
                theta_k = torch.linalg.pinv(A) @ b# (6, 2)
                # Reconstruct flow
                Fk_hat = Ek @ theta_k
                flow_reconstruction += Fk_hat.reshape(-1,2)  # (N, 2)
                reconstruction_loss = self.each_mask_criterion(Fk_hat,Fk)
                total_loss += reconstruction_loss*self.each_mask_item_gradient
            reconstruction_loss = self.sum_mask_criterion(pyramid_flow_b, flow_reconstruction)
            total_loss += reconstruction_loss*self.sum_mask_item_gradient

            # Compute reconstruction loss
            # with torch.no_grad():
            #     flow_reconstruction = flow_reconstruction.detach()
            # reconstruction_loss = torch.pow(torch.log((scene_flow_b+1e8)/(flow_reconstruction+1e8)), 2).mean()
        
        # Return average loss
        return total_loss / batch_size