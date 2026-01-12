"""Custom visualization component for tiled ensemble predictions.

使用自定义的可视化方式（visualize_heatmap.py的风格）
"""
from pathlib import Path
import logging
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.components.base import Runner
from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT

logger = logging.getLogger(__name__)


def save_original_and_overlay_with_info(
    image_path,
    anomaly_map,
    pred_label,
    pred_score,
    save_dir,
    image_obj=None,
    roi_mask=None,
):
    """保存原图、叠加图、ROI区域和异常区域的可视化结果。
    
    生成包含裁剪后图片、热力图叠加、ROI区域、异常区域的可视化结果。
    
    Args:
        image_path (str | Path): 图像路径（用于生成保存文件名）。
        anomaly_map (torch.Tensor | np.ndarray): 异常热力图。
        pred_label (str): 预测标签（"OK" 或 "NG"）。
        pred_score (float): 预测分数（0-1）。
        save_dir (str | Path): 保存目录。
        image_obj (PIL.Image | None): PIL Image对象（可选）。如果提供，使用此对象而不是从路径读取。
            Defaults to ``None``.
        roi_mask (np.ndarray | None): ROI mask (uint8, 0-255)。如果提供，显示 ROI 区域。
            Defaults to ``None``.
        
    Returns:
        Path: 保存的图像路径。
        
    Example:
        >>> save_path = save_original_and_overlay_with_info(
        ...     image_path="image.jpg",
        ...     anomaly_map=heatmap,
        ...     pred_label="NG",
        ...     pred_score=0.85,
        ...     save_dir="results",
        ...     roi_mask=roi_mask
        ... )
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读原图（如果提供了image_obj，使用它；否则从路径读取）
    if image_obj is not None:
        # 使用传入的PIL Image
        image = np.array(image_obj)
        # PIL默认是RGB，直接使用
    else:
        # 从路径读取
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            raise FileNotFoundError(f"图像文件不存在或损坏: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. anomaly map
    heatmap = anomaly_map.squeeze().cpu().numpy()
    
    # 确保是 2D 数组（处理可能的多余维度）
    while heatmap.ndim > 2:
        logger.debug(f"降维: {heatmap.shape} -> {heatmap[0].shape}")
        heatmap = heatmap[0]
    
    # 使用 cv2.normalize 进行归一化
    heatmap_norm = cv2.normalize(
        heatmap, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    # 确保是单通道 2D 数组（cv2.applyColorMap 要求）
    if heatmap_norm.ndim != 2:
        logger.error(f"heatmap_norm 维度错误: {heatmap_norm.shape}, 期望 2D")
        raise ValueError(f"heatmap_norm must be 2D, got shape {heatmap_norm.shape}")
    
    logger.debug(f"heatmap_norm shape: {heatmap_norm.shape}, dtype: {heatmap_norm.dtype}")
    
    # 应用colormap (输入必须是单通道 uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = cv2.resize(
        heatmap_color, (image.shape[1], image.shape[0])
    )

    # 3. overlay
    overlay = cv2.addWeighted(image, 0.65, heatmap_color, 0.35, 0)

    # 4. 异常区域蒙版图（其他部分黑色覆盖）
    # 使用更严格的阈值来分离异常区域，只保留高异常分数的区域
    threshold = np.percentile(heatmap, 98)  # 取热力图的98分位数作为阈值（只保留前2%最异常的区域）
    anomaly_mask = (heatmap > threshold).astype(np.uint8) * 255
    
    # 使用形态学操作去除小的噪点，保留主要异常区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)  # 开运算去除小噪点
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)  # 闭运算填充小孔
    
    # 扩展mask到3通道
    anomaly_mask_3ch = cv2.resize(anomaly_mask, (image.shape[1], image.shape[0]))
    anomaly_mask_3ch = np.stack([anomaly_mask_3ch] * 3, axis=-1)
    
    # 创建黑色背景，只保留异常区域
    masked_image = np.zeros_like(image)
    masked_image[anomaly_mask_3ch > 0] = image[anomaly_mask_3ch > 0]
    
    # 给异常区域加红色边框
    contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masked_image = cv2.drawContours(masked_image, contours, -1, (255, 0, 0), 3)  # 红色，3像素粗

    # ===== 文本信息 =====
    label_text = pred_label          # 已经是 "NG" / "OK"
    score_value = float(pred_score)  # 已经是 float
    text = f"{label_text} | score = {score_value:.3f}"
    text_color = "red" if label_text == "NG" else "lime"

    # 5. 生成 ROI 区域可视化（如果提供了 ROI mask）
    roi_visualization = None
    if roi_mask is not None:
        # 调整 ROI mask 到图像尺寸
        roi_mask_resized = cv2.resize(roi_mask, (image.shape[1], image.shape[0]))
        
        # 创建 ROI 可视化：在原图上叠加半透明的 ROI 区域
        roi_visualization = image.copy()
        
        # ROI 内部：绿色半透明覆盖
        roi_overlay = np.zeros_like(image)
        roi_overlay[roi_mask_resized > 0] = [0, 255, 0]  # 绿色
        roi_visualization = cv2.addWeighted(roi_visualization, 0.7, roi_overlay, 0.3, 0)
        
        # ROI 边界：绿色轮廓
        contours, _ = cv2.findContours(roi_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_visualization = cv2.drawContours(roi_visualization, contours, -1, (0, 255, 0), 3)

    # 6. 画整张结果图（根据是否有 ROI 决定显示 3 张或 4 张图）
    if roi_visualization is not None:
        # 有 ROI：显示 4 张图
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Cropped Image")
        axes[0].axis("off")
        
        axes[1].imshow(roi_visualization)
        axes[1].set_title("ROI Region")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Anomaly Overlay")
        axes[2].axis("off")

        axes[3].imshow(masked_image)
        axes[3].set_title("Anomaly Region")
        axes[3].axis("off")

        # 在第3张图（Anomaly Overlay）上添加文本
        axes[2].text(
            20,
            80,
            text,
            color=text_color,
            fontsize=16,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=6),
        )
    else:
        # 无 ROI：显示 3 张图（保持原来的风格）
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title("Cropped Image")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Anomaly Overlay")
        axes[1].axis("off")

        axes[2].imshow(masked_image)
        axes[2].set_title("Anomaly Region")
        axes[2].axis("off")

        axes[1].text(
            20,
            80,
            text,
            color=text_color,
            fontsize=16,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=6),
        )

    plt.tight_layout()

    # 5. 保存整张 figure
    filename = Path(image_path).stem
    save_name = f"{filename}_{label_text}_{score_value:.3f}.jpg"
    save_path = save_dir / save_name

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)   # 🔑 非常重要：不弹窗、不占内存

    print(f"Saved result to: {save_path}")
    return save_path


class CustomVisualizationJob(Job):
    """使用自定义可视化方式的Job。
    
    为每个预测结果生成包含原图、热力图叠加、异常区域的可视化图像。
    
    Args:
        predictions (dict): 预测结果字典列表，包含 image_path, anomaly_map, pred_label, pred_score。
        output_dir (Path): 可视化结果保存目录。
    """

    name = "CustomVisualize"

    def __init__(self, predictions: dict, output_dir: Path) -> None:
        """初始化可视化任务。
        
        Args:
            predictions (dict): 预测结果。
            output_dir (Path): 输出目录。
        """
        super().__init__()
        self.predictions = predictions
        self.output_dir = output_dir

    def run(self) -> dict:
        """运行可视化任务。
        
        Returns:
            dict: 原始预测结果（保持不变）。
        """
        logger.info(f"开始可视化，共 {len(self.predictions)} 个预测结果")
        
        for idx, data in enumerate(tqdm(self.predictions, desc="Custom Visualizing")):
            logger.debug(f"处理第 {idx+1} 个图像...")
            
            # 提取数据（使用对象属性访问）
            logger.debug("提取数据...")
            image_path = data.image_path
            anomaly_map = data.anomaly_map
            pred_label = data.pred_label
            pred_score = data.pred_score
            
            logger.debug(f"image_path类型: {type(image_path)}")
            logger.debug(f"anomaly_map shape: {anomaly_map.shape if hasattr(anomaly_map, 'shape') else 'no shape'}")
            
            # 处理image_path（可能是列表）
            if isinstance(image_path, (list, tuple)):
                image_path = image_path[0]
            
            # 转换label
            if isinstance(pred_label, torch.Tensor):
                # 处理不同形状的tensor
                if pred_label.numel() == 1:
                    pred_label = "NG" if pred_label.item() > 0 else "OK"
                else:
                    # 如果有多个元素，取第一个或最大值
                    pred_label = "NG" if pred_label.max().item() > 0 else "OK"
            elif isinstance(pred_label, (int, float)):
                pred_label = "NG" if pred_label > 0 else "OK"
            elif isinstance(pred_label, str):
                pass  # 已经是字符串
            else:
                pred_label = "Unknown"
            
            # 转换score
            if isinstance(pred_score, torch.Tensor):
                # 处理不同形状的tensor
                if pred_score.numel() == 1:
                    pred_score = pred_score.item()
                else:
                    # 如果有多个元素，取平均值或最大值
                    pred_score = pred_score.mean().item()
            
            # 确保 anomaly_map 是 Tensor（save_original_and_overlay_with_info 需要）
            if not isinstance(anomaly_map, torch.Tensor):
                anomaly_map = torch.from_numpy(np.array(anomaly_map))
            
            # 获取 ROI mask（如果有，使用对象属性访问）
            roi_mask = getattr(data, "roi_mask", None)
            
            # 生成可视化
            try:
                result_path = save_original_and_overlay_with_info(
                    image_path=image_path,
                    anomaly_map=anomaly_map,
                    pred_label=pred_label,
                    pred_score=pred_score,
                    save_dir=self.output_dir,
                    roi_mask=roi_mask,
                )
                logger.info(f"已保存: {result_path}")
            except Exception as e:
                logger.error(f"可视化失败 {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return self.predictions

    @staticmethod
    def collect(results: list) -> GATHERED_RESULTS:
        """收集结果"""
        return results[0] if results else None

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """不需要额外保存"""


class CustomVisualizationJobGenerator(JobGenerator):
    """自定义可视化Job生成器。
    
    Args:
        output_dir (Path): 可视化结果保存目录。
    """

    def __init__(self, output_dir: Path) -> None:
        """初始化生成器。
        
        Args:
            output_dir (Path): 输出目录。
        """
        self.output_dir = Path(output_dir)

    @property
    def job_class(self) -> type:
        """返回 Job 类。
        
        Returns:
            type: CustomVisualizationJob 类。
        """
        return CustomVisualizationJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ):
        """生成可视化job。
        
        Args:
            args (dict | None): 额外参数（未使用）。
            prev_stage_result (PREV_STAGE_RESULT): 前一阶段的预测结果。
            
        Yields:
            CustomVisualizationJob: 可视化任务。
            
        Raises:
            ValueError: 如果没有前一阶段的结果。
        """
        if prev_stage_result is None:
            msg = "Custom Visualization job requires predictions from previous step."
            raise ValueError(msg)

        yield CustomVisualizationJob(prev_stage_result, self.output_dir)

