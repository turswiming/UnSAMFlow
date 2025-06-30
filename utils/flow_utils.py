import os

import cv2
import imageio
import numpy as np
import torch


def load_flow(path):
    if path.endswith(".png"):
        # for KITTI which uses 16bit PNG images
        # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
        # The -1 is here to specify not to change the image depth (16bit), and is compatible
        # with both OpenCV2 and OpenCV3
        flo_file = cv2.imread(path, -1)
        flo_img = flo_file[:, :, 2:0:-1].astype(np.float32)
        invalid = flo_file[:, :, 0] == 0  # mask
        flo_img = flo_img - 32768
        flo_img = flo_img / 64
        flo_img[np.abs(flo_img) < 1e-10] = 1e-10
        flo_img[invalid, :] = 0
        return flo_img, np.expand_dims(flo_file[:, :, 0], 2)
    else:
        with open(path, "rb") as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert 202021.25 == magic, "Magic number incorrect. Invalid .flo file"
            h = np.fromfile(f, np.int32, count=1)[0]
            w = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (w, h, 2))
        return data2D


def load_mask(path):
    # 0~255 HxWx1
    mask = imageio.imread(path).astype(np.float32) / 255.0
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    return np.expand_dims(mask, -1)


def flow_to_image(flow, max_flow=None):
    """
    Convert flow field to color image using the Middlebury color wheel.
    
    Args:
        flow: numpy array of shape (H, W, 2) or (B, H, W, 2)
        max_flow: maximum flow magnitude for normalization. If None, use max in flow.
    
    Returns:
        color_image: numpy array of shape (H, W, 3) or (B, H, W, 3) in uint8
    """
    try:
        if flow.ndim == 4:
            # Batch dimension
            return np.stack([flow_to_image(f, max_flow) for f in flow], axis=0)
        
        # Ensure flow is numpy array
        if isinstance(flow, torch.Tensor):
            flow = flow.detach().cpu().numpy()
        
        # Handle unknown flow values
        UNKNOWN_FLOW_THRESH = 1e9
        unknown_mask = (np.abs(flow[:, :, 0]) > UNKNOWN_FLOW_THRESH) | (np.abs(flow[:, :, 1]) > UNKNOWN_FLOW_THRESH)
        flow_clean = flow.copy()
        flow_clean[unknown_mask] = 0
        
        # Calculate flow magnitude
        u, v = flow_clean[:, :, 0], flow_clean[:, :, 1]
        rad = np.sqrt(u**2 + v**2)
        
        # Normalize flow
        if max_flow is None:
            max_flow = np.max(rad) if np.max(rad) > 0 else 1.0
        else:
            max_flow = max(max_flow, 1.0)
        
        u = u / max_flow
        v = v / max_flow
        rad = rad / max_flow
        
        # Create color wheel
        colorwheel = make_colorwheel()
        ncols = colorwheel.shape[0]
        
        # Calculate angle and map to color wheel
        angle = np.arctan2(-v, -u) / np.pi
        fk = (angle + 1) / 2 * (ncols - 1) + 1
        k0 = np.floor(fk).astype(int)
        k1 = (k0 + 1) % ncols
        f = fk - k0
        
        # Ensure k0 is within bounds
        k0 = np.clip(k0, 0, ncols - 1)
        k1 = np.clip(k1, 0, ncols - 1)
        
        # Interpolate colors
        img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        for i in range(3):
            col0 = colorwheel[k0, i] / 255.0
            col1 = colorwheel[k1, i] / 255.0
            col = (1 - f) * col0 + f * col1
            
            # Increase saturation with radius
            col[rad <= 1] = 1 - rad[rad <= 1] * (1 - col[rad <= 1])
            col[rad > 1] = col[rad > 1] * 0.75  # Out of range
            
            # Fix: unknown_mask is 2D, so we don't need the third index
            img[:, :, i] = np.floor(255 * col * (1 - unknown_mask)).astype(np.uint8)
        
        return img
        
    except Exception as e:
        print(f"Warning: Failed to create flow visualization: {e}")
        # Return a fallback image
        if flow.ndim == 4:
            # Batch case
            return np.zeros((flow.shape[0], flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
        else:
            # Single image case
            return np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)


def make_colorwheel():
    """
    Create the Middlebury color wheel for flow visualization.
    
    Returns:
        colorwheel: numpy array of shape (55, 3) with RGB values
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)
    
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col = col + RY
    
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col = col + YG
    
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col = col + GC
    
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col = col + CB
    
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(BM) / BM)
    col = col + BM
    
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col+MR, 0] = 255
    
    return colorwheel


def resize_flow(flow, new_shape):
    _, _, h, w = flow.shape
    new_h, new_w = new_shape
    flow = torch.nn.functional.interpolate(
        flow, (new_h, new_w), mode="bilinear", align_corners=True
    )
    scale_h, scale_w = h / float(new_h), w / float(new_w)
    flow[:, 0] /= scale_w
    flow[:, 1] /= scale_h
    return flow


def tensor2array(tensor, max_value=None, colormap='rainbow', channel_first=True):
    """
    Convert tensor to numpy array for visualization.
    
    Args:
        tensor: torch.Tensor
        max_value: maximum value for normalization
        colormap: colormap for visualization
        channel_first: whether tensor is in channel-first format
    
    Returns:
        array: numpy array ready for visualization
    """
    tensor = tensor.detach().cpu().numpy()
    if channel_first:
        tensor = tensor.transpose(0, 2, 3, 1)
    
    if max_value is None:
        max_value = np.max(tensor)
    
    if tensor.shape[-1] == 1:
        # Single channel - apply colormap
        import matplotlib.pyplot as plt
        cm = plt.get_cmap(colormap)
        # Handle the case where tensor might be 2D or 3D
        if tensor.ndim == 3:
            # Single image: (H, W, 1)
            colored = cm(tensor[:, :, 0] / max_value)[:, :, :3]
            tensor = (colored * 255).astype(np.uint8)
        else:
            # Batch of images: (B, H, W, 1)
            colored = cm(tensor[:, :, :, 0] / max_value)[:, :, :, :3]
            tensor = (colored * 255).astype(np.uint8)
    elif tensor.shape[-1] == 2:
        # Flow field
        tensor = flow_to_image(tensor, max_value)
    elif tensor.shape[-1] == 3:
        # RGB image
        tensor = np.clip(tensor, 0, 1)
        tensor = (tensor * 255).astype(np.uint8)
    
    return tensor


def create_visualization_grid(img1, flow, mask, max_flow=None, max_images=4):
    """
    Create a grid visualization combining image, flow, and mask.
    
    Args:
        img1: torch.Tensor of shape (B, 3, H, W) - first image
        flow: torch.Tensor of shape (B, 2, H, W) - optical flow
        mask: torch.Tensor of shape (B, C, H, W) - segmentation mask
        max_flow: maximum flow magnitude for normalization
        max_images: maximum number of images to visualize
    
    Returns:
        grid: numpy array of shape (H*3, W*max_images, 3) - visualization grid
    """
    B = min(img1.shape[0], max_images)
    
    # Convert tensors to numpy arrays
    img1_np = tensor2array(img1[:B], max_value=1.0)
    flow_np = tensor2array(flow[:B], max_value=max_flow)
    
    # Convert mask to visualization
    try:
        if mask.shape[1] > 1:  # Multi-class mask
            mask_argmax = torch.argmax(mask[:B], dim=1, keepdim=True)
            mask_np = tensor2array(mask_argmax, max_value=mask.shape[1]-1, colormap='tab20')
        else:  # Binary mask
            mask_np = tensor2array(mask[:B], max_value=1.0, colormap='gray')
    except Exception as e:
        print(f"Warning: Failed to process mask in create_visualization_grid: {e}")
        # Create a fallback mask visualization
        mask_np = np.zeros((B, img1_np.shape[1], img1_np.shape[2], 3), dtype=np.uint8)
    
    # Create grid
    rows = []
    for i in range(B):
        try:
            row = np.concatenate([img1_np[i], flow_np[i], mask_np[i]], axis=1)
            rows.append(row)
        except Exception as e:
            print(f"Warning: Failed to create row {i} in visualization grid: {e}")
            # Create empty row as fallback
            empty_row = np.zeros((img1_np.shape[1], img1_np.shape[2] * 3, 3), dtype=np.uint8)
            rows.append(empty_row)
    
    # Pad with empty images if needed
    while len(rows) < max_images:
        empty_img = np.zeros_like(rows[0])
        rows.append(empty_img)
    
    grid = np.concatenate(rows, axis=0)
    return grid


# credit: https://github.com/princeton-vl/RAFT/blob/master/core/utils/frame_utils.py
def writeFlowSintel(filename, uv, v=None):
    """Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2
    TAG_CHAR = np.array([202021.25], np.float32)

    if v is None:
        assert uv.ndim == 3
        assert uv.shape[2] == 2
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert u.shape == v.shape
    height, width = u.shape

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        # write the header
        f.write(TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width * nBands))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)


# credit: https://github.com/princeton-vl/RAFT/blob/master/core/utils/frame_utils.py
def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def evaluate_flow(gt_flows, pred_flows, moving_masks=None):
    # credit "undepthflow/eval/evaluate_flow.py"
    def calculate_error_rate(epe_map, gt_flow, mask):
        bad_pixels = np.logical_and(
            epe_map * mask > 3,
            epe_map * mask > 0.05 * np.sqrt(np.sum(np.square(gt_flow), axis=2)),
        )
        return bad_pixels.sum() / mask.sum() * 100.0

    (
        error,
        error_noc,
        error_occ,
        error_move,
        error_static,
        error_rate,
        error_rate_noc,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    error_move_rate, error_static_rate = 0.0, 0.0
    B = len(gt_flows)
    for gt_flow, pred_flow, i in zip(gt_flows, pred_flows, range(B)):
        H, W = gt_flow.shape[:2]
        h, w = pred_flow.shape[:2]

        # pred_flow = np.copy(pred_flow)
        # pred_flow[:, :, 0] = pred_flow[:, :, 0] / w * W
        # pred_flow[:, :, 1] = pred_flow[:, :, 1] / h * H

        # flo_pred = cv2.resize(pred_flow, (W, H), interpolation=cv2.INTER_LINEAR)

        pred_flow = torch.from_numpy(pred_flow)[None].permute(0, 3, 1, 2)
        flo_pred = resize_flow(pred_flow, (H, W))
        flo_pred = flo_pred[0].numpy().transpose(1, 2, 0)

        epe_map = np.sqrt(
            np.sum(np.square(flo_pred[:, :, :2] - gt_flow[:, :, :2]), axis=2)
        )
        if gt_flow.shape[-1] == 2:
            error += np.mean(epe_map)

        elif gt_flow.shape[-1] == 4:  # with occ and noc mask
            error += np.sum(epe_map * gt_flow[:, :, 2]) / np.sum(gt_flow[:, :, 2])
            noc_mask = gt_flow[:, :, -1]
            error_noc += np.sum(epe_map * noc_mask) / np.sum(noc_mask)

            error_occ += np.sum(epe_map * (gt_flow[:, :, 2] - noc_mask)) / max(
                np.sum(gt_flow[:, :, 2] - noc_mask), 1.0
            )

            error_rate += calculate_error_rate(
                epe_map, gt_flow[:, :, 0:2], gt_flow[:, :, 2]
            )
            error_rate_noc += calculate_error_rate(
                epe_map, gt_flow[:, :, 0:2], noc_mask
            )
            if moving_masks is not None:
                move_mask = moving_masks[i]

                error_move_rate += calculate_error_rate(
                    epe_map, gt_flow[:, :, 0:2], gt_flow[:, :, 2] * move_mask
                )
                error_static_rate += calculate_error_rate(
                    epe_map, gt_flow[:, :, 0:2], gt_flow[:, :, 2] * (1.0 - move_mask)
                )

                error_move += np.sum(epe_map * gt_flow[:, :, 2] * move_mask) / np.sum(
                    gt_flow[:, :, 2] * move_mask
                )
                error_static += np.sum(
                    epe_map * gt_flow[:, :, 2] * (1.0 - move_mask)
                ) / np.sum(gt_flow[:, :, 2] * (1.0 - move_mask))

    if gt_flows[0].shape[-1] == 4:
        res = [
            error / B,
            error_noc / B,
            error_occ / B,
            error_rate / B,
            error_rate_noc / B,
        ]
        if moving_masks is not None:
            res += [error_move / B, error_static / B]
        return res
    else:
        return [error / B]
