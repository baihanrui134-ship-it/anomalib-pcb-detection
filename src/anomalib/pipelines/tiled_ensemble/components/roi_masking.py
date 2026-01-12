"""ROI Mask processing component for tiled ensemble predictions.

应用 ROI mask 过滤异常分数，只计算感兴趣区域内的异常。
"""
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT

logger = logging.getLogger(__name__)


def load_roi_mask_from_json(json_path: Path, image_shape: tuple[int, int]) -> np.ndarray | None:
    """从 JSON 文件加载 ROI mask。
    
    支持多种 JSON 格式：
    1. 单个多边形: {"polygon": [[x1, y1], [x2, y2], ...]}
    2. 单个矩形: {"bbox": [x, y, width, height]}
    3. 多个 ROI: {"rois": [{"x": x, "y": y, "width": w, "height": h}, ...]}
    
    Args:
        json_path (Path): ROI JSON 文件路径。
        image_shape (tuple[int, int]): 图像尺寸 (height, width)。
        
    Returns:
        np.ndarray | None: ROI mask (uint8, 0-255)，如果失败返回 None。
    """
    if not json_path.exists():
        logger.warning(f"ROI 文件不存在: {json_path}")
        return None
    
    try:
        with open(json_path, "r") as f:
            roi_data = json.load(f)
        
        # 创建空白 mask
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 解析不同格式的 ROI
        if "polygon" in roi_data:
            # 格式1: 单个多边形 ROI
            points = np.array(roi_data["polygon"], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
            logger.info(f"成功加载多边形 ROI: {json_path}")
        elif "bbox" in roi_data:
            # 格式2: 单个矩形 ROI (bbox格式: [x, y, width, height])
            x, y, w_roi, h_roi = roi_data["bbox"]
            mask[y:y+h_roi, x:x+w_roi] = 255
            logger.info(f"成功加载矩形 ROI: {json_path}")
        elif "rois" in roi_data:
            # 格式3: 多个矩形 ROI (支持批量处理)
            rois = roi_data["rois"]
            for idx, roi in enumerate(rois):
                x = roi.get("x", 0)
                y = roi.get("y", 0)
                w_roi = roi.get("width", 0)
                h_roi = roi.get("height", 0)
                
                # 填充当前 ROI 区域
                mask[y:y+h_roi, x:x+w_roi] = 255
            
            logger.info(f"成功加载 {len(rois)} 个 ROI 区域: {json_path}")
        else:
            logger.error(f"ROI 格式不支持，需要包含 'polygon'、'bbox' 或 'rois' 字段: {json_path}")
            return None
        
        return mask
    
    except Exception as e:
        logger.error(f"加载 ROI mask 失败 {json_path}: {e}")
        return None


class ROIMaskJob(Job):
    """ROI Mask 应用任务。
    
    对预测结果应用 ROI mask，只保留感兴趣区域内的异常分数。
    
    Args:
        predictions (dict): 预测结果字典列表。
        roi_dir (Path | None): ROI JSON 文件目录。如果为 None，则不应用 ROI。
        default_roi_file (str | None): 通用 ROI 文件名。如果指定，所有图片使用该 ROI 文件。
    """

    name = "ROIMask"

    def __init__(
        self,
        predictions: dict,
        roi_dir: Path | None = None,
        default_roi_file: str | None = None,
    ) -> None:
        """初始化 ROI mask 任务。
        
        Args:
            predictions (dict): 预测结果。
            roi_dir (Path | None): ROI 目录。
            default_roi_file (str | None): 通用 ROI 文件名（批量处理时使用）。
        """
        super().__init__()
        self.predictions = predictions
        self.roi_dir = Path(roi_dir) if roi_dir else None
        self.default_roi_file = default_roi_file

    def run(self) -> dict:
        """运行 ROI mask 应用。
        
        Returns:
            dict: 应用 ROI mask 后的预测结果。
        """
        if self.roi_dir is None or not self.roi_dir.exists():
            logger.info("未启用 ROI mask 或 ROI 目录不存在，跳过 ROI 处理")
            return self.predictions
        
        # 判断是否使用通用 ROI 文件
        use_default_roi = self.default_roi_file is not None
        default_roi_path = None
        default_roi_mask_cache = None  # 缓存通用 ROI mask
        
        if use_default_roi:
            default_roi_path = self.roi_dir / self.default_roi_file
            if not default_roi_path.exists():
                logger.error(f"通用 ROI 文件不存在: {default_roi_path}，将跳过 ROI 处理")
                return self.predictions
            logger.info(f"开始应用通用 ROI mask: {default_roi_path}")
        else:
            logger.info(f"开始应用 ROI mask（每张图片独立 ROI），ROI 目录: {self.roi_dir}")
        
        for idx, data in enumerate(self.predictions):
            # 获取图像路径（使用对象属性访问）
            image_path = data.image_path
            if isinstance(image_path, (list, tuple)):
                image_path = image_path[0]
            image_path = Path(image_path)
            
            # 确定使用哪个 ROI 文件
            if use_default_roi:
                roi_json = default_roi_path
            else:
                # 传统模式：为每张图片查找对应的 ROI 文件
                roi_json = self.roi_dir / f"{image_path.stem}_roi.json"
                
                if not roi_json.exists():
                    logger.debug(f"跳过（无 ROI 文件）: {image_path.name}")
                    continue
            
            # 获取 anomaly map（使用对象属性访问）
            anomaly_map = data.anomaly_map
            if isinstance(anomaly_map, torch.Tensor):
                anomaly_map_np = anomaly_map.squeeze().cpu().numpy()
            else:
                anomaly_map_np = np.array(anomaly_map).squeeze()
            
            # 确保是 2D
            while anomaly_map_np.ndim > 2:
                anomaly_map_np = anomaly_map_np[0]
            
            # 加载 ROI mask（通用 ROI 使用缓存）
            if use_default_roi and default_roi_mask_cache is not None:
                # 检查缓存的 mask 尺寸是否匹配
                if default_roi_mask_cache.shape == anomaly_map_np.shape:
                    roi_mask = default_roi_mask_cache
                else:
                    # 尺寸不匹配，重新加载
                    roi_mask = load_roi_mask_from_json(roi_json, anomaly_map_np.shape)
                    default_roi_mask_cache = roi_mask
            else:
                roi_mask = load_roi_mask_from_json(roi_json, anomaly_map_np.shape)
                if use_default_roi:
                    default_roi_mask_cache = roi_mask
            
            if roi_mask is None:
                continue
            
            # 应用 ROI mask
            roi_mask_normalized = roi_mask / 255.0
            masked_anomaly_map = anomaly_map_np * roi_mask_normalized
            
            # 重新计算 pred_score（只基于 ROI 内区域）
            roi_area = roi_mask_normalized.sum()
            if roi_area > 0:
                new_pred_score = masked_anomaly_map.sum() / roi_area
            else:
                new_pred_score = 0.0
            
            # 更新数据（使用对象属性赋值）
            if isinstance(data.anomaly_map, torch.Tensor):
                data.anomaly_map = torch.from_numpy(masked_anomaly_map).unsqueeze(0)
            else:
                data.anomaly_map = masked_anomaly_map
            
            # 安全提取旧分数（处理 tensor 情况）
            if isinstance(data.pred_score, torch.Tensor):
                old_score = data.pred_score.item() if data.pred_score.numel() == 1 else data.pred_score.mean().item()
            else:
                old_score = float(data.pred_score)
            
            # 更新 pred_score（转换为 tensor，保持一致）
            data.pred_score = torch.tensor(new_pred_score)
            
            # 根据新分数更新标签（使用 tensor，与 thresholding 组件保持一致）
            # pred_label: 0.0 = OK, 1.0 = NG
            data.pred_label = torch.tensor(1.0 if new_pred_score > 0.5 else 0.0)
            
            # 💾 保存 ROI mask 到数据中（用于可视化）
            data.roi_mask = roi_mask  # 保存 ROI mask (uint8, 0-255)
            
            logger.info(f"✓ 应用 ROI: {image_path.name} | "
                       f"score: {old_score:.3f} → {new_pred_score:.3f}")
        
        return self.predictions

    @staticmethod
    def collect(results: list) -> GATHERED_RESULTS:
        """收集结果。
        
        Args:
            results (list): 结果列表。
            
        Returns:
            GATHERED_RESULTS: 收集的结果。
        """
        return results[0] if results else None

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """不需要额外保存。
        
        Args:
            results (GATHERED_RESULTS): 结果。
        """


class ROIMaskJobGenerator(JobGenerator):
    """ROI Mask Job 生成器。
    
    Args:
        roi_dir (Path | None): ROI JSON 文件目录。
        default_roi_file (str | None): 通用 ROI 文件名（用于批量处理所有图片）。
    """

    def __init__(self, roi_dir: Path | None = None, default_roi_file: str | None = None) -> None:
        """初始化生成器。
        
        Args:
            roi_dir (Path | None): ROI 目录。
            default_roi_file (str | None): 通用 ROI 文件名（如果指定，所有图片使用同一个 ROI）。
        """
        self.roi_dir = Path(roi_dir) if roi_dir else None
        self.default_roi_file = default_roi_file

    @property
    def job_class(self) -> type:
        """返回 Job 类。
        
        Returns:
            type: ROIMaskJob 类。
        """
        return ROIMaskJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ):
        """生成 ROI mask job。
        
        Args:
            args (dict | None): 额外参数（未使用）。
            prev_stage_result (PREV_STAGE_RESULT): 前一阶段的预测结果。
            
        Yields:
            ROIMaskJob: ROI mask 任务。
            
        Raises:
            ValueError: 如果没有前一阶段的结果。
        """
        if prev_stage_result is None:
            msg = "ROI Mask job requires predictions from previous step."
            raise ValueError(msg)

        yield ROIMaskJob(prev_stage_result, self.roi_dir, self.default_roi_file)

