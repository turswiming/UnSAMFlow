"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from . import kitti_trainer_ar, sintel_trainer_ar, sintel_trainer_ar_2model


def get_trainer(name):
    if name == "KITTI_AR":
        TrainFramework = kitti_trainer_ar.TrainFramework
    elif name == "SINTEL_AR":
        TrainFramework = sintel_trainer_ar.TrainFramework
    elif name == "SINTEL_AR_2MODEL":
        TrainFramework = sintel_trainer_ar_2model.TrainFramework 
    else:
        raise NotImplementedError(name)

    return TrainFramework
