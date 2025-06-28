from .pwclite import PWCLite
from .swin_unet import SwinUNet

def get_model(cfg):
    if cfg.type == "pwclite":
        model = PWCLite(cfg)
    elif cfg.type == "swin_unet":
        # Get Swin-UNet configuration from cfg
        img_size = getattr(cfg, 'img_size', 224)
        in_channels = getattr(cfg, 'in_channels', 3)
        out_channels = getattr(cfg, 'out_channels', 2)
        embed_dim = getattr(cfg, 'embed_dim', 96)
        depths = getattr(cfg, 'depths', [2, 2, 6, 2])
        num_heads = getattr(cfg, 'num_heads', [3, 6, 12, 24])
        window_size = getattr(cfg, 'window_size', 7)
        mlp_ratio = getattr(cfg, 'mlp_ratio', 4.0)
        qkv_bias = getattr(cfg, 'qkv_bias', True)
        qk_scale = getattr(cfg, 'qk_scale', None)
        drop_rate = getattr(cfg, 'drop_rate', 0.0)
        attn_drop_rate = getattr(cfg, 'attn_drop_rate', 0.0)
        drop_path_rate = getattr(cfg, 'drop_path_rate', 0.1)
        use_checkpoint = getattr(cfg, 'use_checkpoint', False)
        
        model = SwinUNet(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint
        )
    else:
        raise NotImplementedError(cfg.type)
    return model

def get_mask_model(cfg):
    if hasattr(cfg, 'type') and cfg.type == "swin_unet":
        # Get Swin-UNet configuration from cfg
        img_size = getattr(cfg, 'img_size', 224)
        in_channels = getattr(cfg, 'in_channels', 3)
        out_channels = getattr(cfg, 'out_channels', 20)  # Default to 20 for mask model
        embed_dim = getattr(cfg, 'embed_dim', 96)
        depths = getattr(cfg, 'depths', [2, 2, 6, 2])
        num_heads = getattr(cfg, 'num_heads', [3, 6, 12, 24])
        window_size = getattr(cfg, 'window_size', 7)
        mlp_ratio = getattr(cfg, 'mlp_ratio', 4.0)
        qkv_bias = getattr(cfg, 'qkv_bias', True)
        qk_scale = getattr(cfg, 'qk_scale', None)
        drop_rate = getattr(cfg, 'drop_rate', 0.0)
        attn_drop_rate = getattr(cfg, 'attn_drop_rate', 0.0)
        drop_path_rate = getattr(cfg, 'drop_path_rate', 0.1)
        use_checkpoint = getattr(cfg, 'use_checkpoint', False)
        
        return SwinUNet(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint
        )
    else:
        # Default to original MaskUNet
        raise NotImplementedError(cfg.type)