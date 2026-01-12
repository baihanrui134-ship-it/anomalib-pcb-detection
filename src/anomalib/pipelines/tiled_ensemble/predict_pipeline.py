# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble prediction pipeline for new images.

This pipeline is used to predict anomalies on new images (without ground truth).
Unlike EvalTiledEnsemble which is for test set with ground truth and metrics calculation,
this pipeline is for production inference on new images.
"""

import logging
from pathlib import Path

import torch

from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.runners import ParallelRunner, SerialRunner
from anomalib.pipelines.tiled_ensemble.components import (
    MergeJobGenerator,
    NormalizationJobGenerator,
    PredictJobGenerator,
    SmoothingJobGenerator,
    ThresholdingJobGenerator,
    VisualizationJobGenerator,
)
from anomalib.pipelines.tiled_ensemble.components.custom_visualization import CustomVisualizationJobGenerator
from anomalib.pipelines.tiled_ensemble.components.roi_masking import ROIMaskJobGenerator
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage, PredictData, ThresholdingStage

logger = logging.getLogger(__name__)


class PredictTiledEnsemble(Pipeline):
    """Tiled ensemble prediction pipeline for new images.

    This pipeline performs anomaly prediction on new images without ground truth.
    
    Pipeline stages:
        1. Prediction on new images using trained models
        2. Merging tile-level predictions to image-level
        3. (Optional) Seam smoothing
        4. (Optional) Normalization
        5. (Optional) Thresholding
        6. Visualization and save results

    Args:
        root_dir (Path): Path to root directory containing trained model checkpoints.

    Example:
        >>> from pathlib import Path
        >>> pipeline = PredictTiledEnsemble(root_dir=Path("results/PatchCore/pcb/latest"))
        >>> pipeline.run()
    """

    def __init__(self, root_dir: Path | str) -> None:
        """Initialize prediction pipeline.
        
        Args:
            root_dir (Path | str): Path to trained model directory containing checkpoints.
        """
        self.root_dir = Path(root_dir)
        self.output_dir = self.root_dir  # Default to root_dir, can be overridden in _setup_runners
        
        # Validate that checkpoint directory exists
        checkpoint_dir = self.root_dir / "weights" / "lightning"
        if not checkpoint_dir.exists():
            msg = f"找不到Checkpoint目录: {checkpoint_dir}"
            raise FileNotFoundError(msg)
        
        logger.info(f"预测pipeline已初始化，模型来自: {self.root_dir}")

    def _setup_runners(self, args: dict) -> list[Runner]:
        """Set up the runners for the prediction pipeline.

        This pipeline consists of:
        Prediction on new images > merging of predictions > (optional) seam smoothing
        > (optional) normalization > (optional) thresholding > visualization of predictions

        Args:
            args (dict): Pipeline configuration arguments.

        Returns:
            list[Runner]: List of runners executing tiled ensemble prediction jobs.
        """
        runners: list[Runner] = []

        seed = args.get("seed", 42)
        accelerator = args.get("accelerator", "cuda")
        tiling_args = args["tiling"]
        data_args = args["data"]
        normalization_stage = NormalizationStage(args.get("normalization_stage", "image"))
        thresholding_stage = ThresholdingStage(args.get("thresholding_stage", "image"))
        model_args = args["TrainModels"]["model"]
        
        # Get output directory from config (if specified)
        output_dir = args.get("output_dir", None)
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            logger.info(f"自定义输出目录: {self.output_dir}")
        else:
            self.output_dir = self.root_dir
            logger.info(f"使用默认输出目录: {self.output_dir}")

        # Configure prediction job for new images (use TEST split mode)
        predict_job_generator = PredictJobGenerator(
            PredictData.TEST,  # Use TEST mode for prediction
            seed=seed,
            accelerator=accelerator,
            root_dir=self.root_dir,
            tiling_args=tiling_args,
            data_args=data_args,
            model_args=model_args,
            normalization_stage=normalization_stage,
        )

        # 1. Predict using new images
        logger.info("正在设置预测任务...")
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if accelerator == "cuda" and n_gpus > 0:
            logger.info(f"使用 {n_gpus} 个GPU进行并行预测")
            runners.append(
                ParallelRunner(
                    predict_job_generator,
                    n_jobs=n_gpus,
                ),
            )
        else:
            logger.info("使用串行模式进行预测")
            runners.append(
                SerialRunner(
                    predict_job_generator,
                ),
            )

        # 2. Merge tile predictions into image-level predictions
        logger.info("正在设置合并任务...")
        runners.append(SerialRunner(MergeJobGenerator(tiling_args=tiling_args, data_args=data_args)))

        # 3. (Optional) Smooth seams
        if args.get("SeamSmoothing", {}).get("apply", True):
            logger.info("正在设置接缝平滑任务...")
            runners.append(
                SerialRunner(
                    SmoothingJobGenerator(accelerator=accelerator, tiling_args=tiling_args, data_args=data_args),
                ),
            )

        # 4. (Optional) Normalize predictions
        if normalization_stage == NormalizationStage.IMAGE:
            logger.info("正在设置归一化任务...")
            runners.append(SerialRunner(NormalizationJobGenerator(self.root_dir)))

        # 5. (Optional) Apply threshold to get labels from scores
        if thresholding_stage == ThresholdingStage.IMAGE:
            logger.info("正在设置阈值判断任务...")
            runners.append(SerialRunner(ThresholdingJobGenerator(self.root_dir, normalization_stage)))

        # 5.5. (Optional) Apply ROI mask to filter anomaly scores
        roi_dir = args.get("roi_dir", None)
        default_roi_file = args.get("default_roi_file", None)
        
        if roi_dir is not None:
            if default_roi_file:
                logger.info(f"正在设置ROI mask任务（通用ROI模式）... "
                          f"ROI目录: {roi_dir}, 通用ROI文件: {default_roi_file}")
            else:
                logger.info(f"正在设置ROI mask任务（单独ROI模式）... ROI目录: {roi_dir}")
            
            runners.append(SerialRunner(ROIMaskJobGenerator(roi_dir=roi_dir, default_roi_file=default_roi_file)))
        else:
            logger.info("未启用ROI mask（roi_dir未配置）")

        # 6. Visualize predictions (使用自定义可视化)
        logger.info("正在设置自定义可视化任务...")
        runners.append(
            SerialRunner(
                CustomVisualizationJobGenerator(
                    output_dir=self.output_dir,
                )
            ),
        )

        logger.info(f"预测pipeline已配置完成，共{len(runners)}个阶段。")
        return runners

