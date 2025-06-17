from .pwclite import PWCLite
from .UNet import MaskUNet

def get_model(cfg):
    if cfg.type == "pwclite":
        model = PWCLite(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return model
def get_mask_model(cfg):
    return MaskUNet(
        n_channels=3,
        n_classes=20,
        n_filters=64,
        bilinear=True,
        use_batchnorm=True
    )