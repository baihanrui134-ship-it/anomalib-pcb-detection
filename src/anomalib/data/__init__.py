# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib Datasets.

This module provides datasets and data modules for anomaly detection tasks.

The module contains:
    - Data classes for representing different types of data (images, videos, etc.)
    - Dataset classes for loading and processing data
    - Data modules for use with PyTorch Lightning
    - Helper functions for data loading and validation

Example:
    >>> from anomalib.data import MVTecAD
    >>> datamodule = MVTecAD(
    ...     root="./datasets/MVTecAD",
    ...     category="bottle",
    ...     image_size=(256, 256)
    ... )
"""

import importlib
import logging
from enum import Enum
from itertools import chain

from omegaconf import DictConfig, ListConfig

from anomalib.utils.config import to_tuple

# Dataclasses
from .dataclasses import (
    Batch,
    DatasetItem,
    DepthBatch,
    DepthItem,
    ImageBatch,
    ImageItem,
    InferenceBatch,
    NumpyImageBatch,
    NumpyImageItem,
    NumpyVideoBatch,
    NumpyVideoItem,
    VideoBatch,
    VideoItem,
)

# Datamodules
from .datamodules.base import AnomalibDataModule
from .datamodules.depth import ADAM3D, DepthDataFormat, Folder3D, MVTec3D
from .datamodules.image import (
    BMAD,
    MPDD,
    VAD,
    BTech,
    Datumaro,
    Folder,
    ImageDataFormat,
    Kolektor,
    MVTec,
    MVTecAD,
    MVTecAD2,
    MVTecLOCO,
    RealIAD,
    Tabular,
    Visa,
)
from .datamodules.video import Avenue, ShanghaiTech, UCSDped, VideoDataFormat

# Datasets
from .datasets import AnomalibDataset
from .datasets.depth import ADAM3DDataset, Folder3DDataset, MVTec3DDataset
from .datasets.image import (
    BMADDataset,
    BTechDataset,
    DatumaroDataset,
    FolderDataset,
    KolektorDataset,
    MPDDDataset,
    MVTecADDataset,
    MVTecLOCODataset,
    TabularDataset,
    VADDataset,
    VisaDataset,
)
from .datasets.video import AvenueDataset, ShanghaiTechDataset, UCSDpedDataset
from .predict import PredictDataset

logger = logging.getLogger(__name__)


DataFormat = Enum(  # type: ignore[misc]
    "DataFormat",
    {i.name: i.value for i in chain(DepthDataFormat, ImageDataFormat, VideoDataFormat)},
)


class UnknownDatamoduleError(ModuleNotFoundError):
    """Raised when a datamodule cannot be found."""


def _instantiate_transforms(init_args: dict, key: str) -> None:
    """Instantiate transforms from Lightning CLI format (class_path + init_args).
    
    Args:
        init_args (dict): Dictionary containing initialization arguments.
        key (str): Key for the transforms field (e.g., 'train_augmentations').
    """
    from importlib import import_module
    from omegaconf import DictConfig, ListConfig
    from torchvision.transforms.v2 import Compose
    
    if key not in init_args or init_args[key] is None:
        return
    
    transforms_config = init_args[key]
    
    # If it's already an instantiated Transform object, skip
    # Check for both plain Python types and OmegaConf types
    if not isinstance(transforms_config, (list, dict, ListConfig, DictConfig)):
        return
    
    # Handle list of transforms (both plain list and ListConfig)
    if isinstance(transforms_config, (list, ListConfig)):
        instantiated_transforms = []
        for transform_config in transforms_config:
            if isinstance(transform_config, (dict, DictConfig)) and "class_path" in transform_config:
                # This is a Lightning CLI format config
                class_path = transform_config["class_path"]
                transform_init_args = transform_config.get("init_args", {})
                
                # Import and instantiate the transform class
                try:
                    # Split module path and class name
                    if "." in class_path:
                        module_path = ".".join(class_path.split(".")[:-1])
                        class_name = class_path.split(".")[-1]
                        module = import_module(module_path)
                    else:
                        # If no module path, assume it's from torchvision.transforms.v2
                        module = import_module("torchvision.transforms.v2")
                        class_name = class_path
                    
                    transform_class = getattr(module, class_name)
                    transform_instance = transform_class(**transform_init_args)
                    instantiated_transforms.append(transform_instance)
                    logger.info(f"成功实例化transform: {class_path}")
                except Exception as e:
                    logger.error(f"无法实例化transform {class_path}: {e}")
                    raise
            else:
                # Already instantiated or other format
                instantiated_transforms.append(transform_config)
        
        # Wrap multiple transforms in Compose
        if len(instantiated_transforms) > 1:
            init_args[key] = Compose(instantiated_transforms)
        elif len(instantiated_transforms) == 1:
            init_args[key] = instantiated_transforms[0]
        else:
            init_args[key] = None
    
    # Handle single transform (both dict and DictConfig)
    elif isinstance(transforms_config, (dict, DictConfig)) and "class_path" in transforms_config:
        class_path = transforms_config["class_path"]
        transform_init_args = transforms_config.get("init_args", {})
        
        try:
            if "." in class_path:
                module_path = ".".join(class_path.split(".")[:-1])
                class_name = class_path.split(".")[-1]
                module = import_module(module_path)
            else:
                module = import_module("torchvision.transforms.v2")
                class_name = class_path
            
            transform_class = getattr(module, class_name)
            init_args[key] = transform_class(**transform_init_args)
            logger.info(f"成功实例化transform: {class_path}")
        except Exception as e:
            logger.error(f"无法实例化transform {class_path}: {e}")
            raise


def get_datamodule(config: DictConfig | ListConfig | dict) -> AnomalibDataModule:
    """Get Anomaly Datamodule from config.

    Args:
        config: Configuration for the anomaly model. Can be either:
            - DictConfig from OmegaConf
            - ListConfig from OmegaConf
            - Python dictionary

    Returns:
        PyTorch Lightning DataModule configured according to the input.

    Raises:
        UnknownDatamoduleError: If the specified datamodule cannot be found.

    Example:
        >>> from omegaconf import DictConfig
        >>> config = DictConfig({
        ...     "data": {
        ...         "class_path": "MVTecAD",
        ...         "init_args": {"root": "./datasets/MVTec"}
        ...     }
        ... })
        >>> datamodule = get_datamodule(config)
    """
    logger.info("Loading the datamodule and dataset class from the config.")

    # Getting the datamodule class from the config.
    if isinstance(config, dict):
        config = DictConfig(config)
    config_ = config.data if "data" in config else config

    # All the sub data modules are imported to anomalib.data. So need to import the module dynamically using paths.
    module = importlib.import_module("anomalib.data")
    data_class_name = config_.class_path.split(".")[-1]
    # check if the data_class exists in the module
    if not hasattr(module, data_class_name):
        logger.error(
            f"Dataclass '{data_class_name}' not found in module '{module.__name__}'. "
            f"Available classes are {AnomalibDataModule.__subclasses__()}",
        )
        error_str = f"Dataclass '{data_class_name}' not found in module '{module.__name__}'."
        raise UnknownDatamoduleError(error_str)
    dataclass = getattr(module, data_class_name)

    init_args = {**config_.get("init_args", {})}  # get dict
    if "image_size" in init_args:
        init_args["image_size"] = to_tuple(init_args["image_size"])
    
    # Instantiate transforms if they are in Lightning CLI format (class_path + init_args)
    _instantiate_transforms(init_args, "train_augmentations")
    _instantiate_transforms(init_args, "val_augmentations")
    _instantiate_transforms(init_args, "test_augmentations")
    _instantiate_transforms(init_args, "augmentations")
    
    return dataclass(**init_args)


__all__ = [
    # Base Classes
    "AnomalibDataModule",
    "AnomalibDataset",
    # Data Classes
    "Batch",
    "DatasetItem",
    "DepthBatch",
    "DepthItem",
    "ImageBatch",
    "ImageItem",
    "InferenceBatch",
    "NumpyImageBatch",
    "NumpyImageItem",
    "NumpyVideoBatch",
    "NumpyVideoItem",
    "VideoBatch",
    "VideoItem",
    # Data Formats
    "DataFormat",
    "DepthDataFormat",
    "ImageDataFormat",
    "VideoDataFormat",
    # Depth Data Modules
    "Folder3D",
    "MVTec3D",
    "ADAM3D",
    # Image Data Modules
    "BMAD",
    "BTech",
    "Datumaro",
    "Folder",
    "Kolektor",
    "MPDD",
    "MVTec",  # Include MVTec for backward compatibility
    "MVTecAD",
    "MVTecAD2",
    "MVTecLOCO",
    "RealIAD",
    "Tabular",
    "VAD",
    "Visa",
    # Video Data Modules
    "Avenue",
    "ShanghaiTech",
    "UCSDped",
    # Datasets
    "Folder3DDataset",
    "MVTec3DDataset",
    "BTechDataset",
    "DatumaroDataset",
    "FolderDataset",
    "KolektorDataset",
    "MPDDDataset",
    "ADAM3DDataset",
    "MVTecADDataset",
    "MVTecLOCODataset",
    "TabularDataset",
    "VADDataset",
    "VisaDataset",
    "AvenueDataset",
    "ShanghaiTechDataset",
    "UCSDpedDataset",
    "PredictDataset",
    "BMADDataset",
    # Functions
    "get_datamodule",
    # Exceptions
    "UnknownDatamoduleError",
]
