"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from datasets.flow_datasets import (
    KITTIFlowEval,
    KITTIFlowMV,
    KITTIRawFile,
    Sintel,
    SintelRaw,
)

from torch.utils.data import ConcatDataset
from torchvision import transforms
from transforms import input_transforms
from transforms.ar_transforms.ap_transforms import get_ap_transforms
from transforms.co_transforms import get_co_transforms


def get_dataset(cfg):

    co_transform = get_co_transforms(aug_args=cfg.data_aug)
    ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None

    if cfg.type == "KITTI_Raw+MV_2stage":

        train_input_transform = transforms.Compose(
            [input_transforms.Zoom(*cfg.train_shape), input_transforms.ArrayToTensor()]
        )
        valid_input_transform = transforms.Compose(
            [input_transforms.Zoom(*cfg.test_shape), input_transforms.ArrayToTensor()]
        )

        train_set_1 = KITTIRawFile(
            cfg.root_raw,
            None,  # Removed full_seg_root
            None,  # Removed key_obj_root
            name="kitti-raw",
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform,
        )
        train_set_2_1 = KITTIFlowMV(
            cfg.root_kitti15,
            None,  # Removed full_seg_root
            None,  # Removed key_obj_root
            name="kitti2015-mv",
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform,
        )
        train_set_2_2 = KITTIFlowMV(
            cfg.root_kitti12,
            None,  # Removed full_seg_root
            None,  # Removed key_obj_root
            name="kitti2012-mv",
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform,
        )
        train_set_2 = ConcatDataset([train_set_2_1, train_set_2_2])
        train_set_2.name = "kitti-mv"

        valid_set_1 = KITTIFlowEval(
            cfg.root_kitti15,
            None,  # Removed full_seg_root
            None,  # Removed key_obj_root
            name="kitti2015",
            input_transform=valid_input_transform,
        )
        valid_set_2 = KITTIFlowEval(
            cfg.root_kitti12,
            None,  # Removed full_seg_root
            None,  # Removed key_obj_root
            name="kitti2012",
            input_transform=valid_input_transform,
        )

        train_sets = [train_set_1, train_set_2]
        train_sets_epoches = [cfg.epoches_raw, cfg.epoches_mv]
        valid_sets = [valid_set_1, valid_set_2]

    elif cfg.type == "Sintel_Raw+ft_2stage":

        train_input_transform = transforms.Compose(
            [input_transforms.Zoom(*cfg.test_shape),input_transforms.ArrayToTensor()]
        )
        valid_input_transform = transforms.Compose(
            [input_transforms.Zoom(*cfg.test_shape), input_transforms.ArrayToTensor()]
        )

        train_set_1 = SintelRaw(
            cfg.root_sintel_raw,
            None,  # Removed full_seg_root
            None,  # Removed key_obj_root
            name="sintel-raw",
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform,
        )
        train_set_2_1 = Sintel(
            cfg.root_sintel,
            None,  # Removed full_seg_root
            None,  # Removed key_obj_root
            name="sintel-clean_" + cfg.train_subsplit,
            dataset_type="clean",
            split="train",
            subsplit=cfg.train_subsplit,
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform,
            with_flow=True,
        )
        train_set_2_2 = Sintel(
            cfg.root_sintel,
            None,  # Removed full_seg_root
            None,  # Removed key_obj_root
            name="sintel-final_" + cfg.train_subsplit,
            dataset_type="final",
            split="train",
            subsplit=cfg.train_subsplit,
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform,
            with_flow=True,
        )
        train_set_canval_1 = Sintel(
            cfg.root_sintel,
            None,  # Removed full_seg_root
            None,  # Removed key_obj_root
            name="sintel-clean_" + cfg.train_subsplit,
            dataset_type="clean",
            split="train",
            subsplit=cfg.train_subsplit,
            with_flow=True,  # for validation
            input_transform=valid_input_transform,
        )
        train_set_canval_2 =Sintel(
            cfg.root_sintel,
            None,  # Removed full_seg_root
            None,  # Removed key_obj_root
            name="sintel-final_" + cfg.train_subsplit,
            dataset_type="final",
            split="train",
            subsplit=cfg.train_subsplit,
            with_flow=True,  # for validation
            input_transform=valid_input_transform,
        )
        train_set_2 = ConcatDataset([train_set_2_1, train_set_2_2])
        train_set_2.name = "sintel_clean+final_" + cfg.train_subsplit

        valid_set_1 = Sintel(
            cfg.root_sintel,
            None,  # Removed full_seg_root
            None,  # Removed key_obj_root
            name="sintel-clean_" + cfg.val_subsplit,
            dataset_type="clean",
            split="train",
            subsplit=cfg.val_subsplit,
            with_flow=True,  # for validation
            input_transform=valid_input_transform,
        )
        valid_set_2 = Sintel(
            cfg.root_sintel,
            None,  # Removed full_seg_root
            None,  # Removed key_obj_root
            name="sintel-final_" + cfg.val_subsplit,
            dataset_type="final",
            split="train",
            subsplit=cfg.val_subsplit,
            with_flow=True,  # for validation
            input_transform=valid_input_transform,
        )

        train_sets_canval = [train_set_canval_1, train_set_canval_2]
        train_sets = [train_set_1, train_set_2]
        # train_sets = [train_set_2]
        train_sets_epoches = [cfg.epoches_raw, cfg.epoches_ft]
        valid_sets = [valid_set_1, valid_set_2]

    return train_sets, valid_sets,train_sets_canval, train_sets_epoches
