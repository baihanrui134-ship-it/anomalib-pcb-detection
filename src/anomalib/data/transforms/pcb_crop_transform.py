"""
PCB Background Cropping Transform

自定义Transform，在数据加载时自动裁剪PCB背景
用于Anomalib训练pipeline
"""

import cv2
import numpy as np
import torch
from torchvision.transforms.v2 import Transform
from torchvision.transforms import v2
from PIL import Image


class PCBBackgroundCrop(Transform):
    """自动裁剪PCB背景的Transform.
    
    在训练时自动检测PCB边界并裁剪背景，只保留PCB主体部分。
    基于crop_pcb.py的检测算法。
    
    Args:
        padding (int): 裁剪时的边距（像素）. Defaults to 10.
        min_area_ratio (float): PCB最小面积占比（用于过滤误检）. Defaults to 0.1.
        return_original_on_fail (bool): 检测失败时是否返回原图. Defaults to True.
        
    Example:
        >>> from torchvision.transforms.v2 import Compose
        >>> transform = Compose([
        ...     PCBBackgroundCrop(padding=10),
        ...     # ... 其他transforms
        ... ])
    """
    
    def __init__(
        self,
        padding: int = 10,
        min_area_ratio: float = 0.1,
        return_original_on_fail: bool = True,
    ):
        super().__init__()
        self.padding = padding
        self.min_area_ratio = min_area_ratio
        self.return_original_on_fail = return_original_on_fail
        
    def _detect_pcb_bounds(self, image_np: np.ndarray) -> tuple[int, int, int, int] | None:
        """检测PCB边界.
        
        Args:
            image_np (np.ndarray): 输入图像 (H, W, C)，BGR格式
            
        Returns:
            tuple[int, int, int, int] | None: (x1, y1, x2, y2) 或 None（检测失败）
        """
        h, w = image_np.shape[:2]
        
        # 转灰度
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # 膨胀和闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找最大轮廓
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # 检查面积是否合理
        if area < (h * w * self.min_area_ratio):
            return None
        
        x, y, w_pcb, h_pcb = cv2.boundingRect(largest)
        
        # 计算裁剪区域（加padding）
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(w, x + w_pcb + self.padding)
        y2 = min(h, y + h_pcb + self.padding)
        
        return (x1, y1, x2, y2)
    
    def _forward(self, *inputs):
        """Transform的核心方法（torchvision v2 API）."""
        # 获取第一个输入（通常是图像）
        image = inputs[0]
        
        # 根据输入类型进行处理
        if isinstance(image, Image.Image):
            # PIL Image -> numpy
            image_np = np.array(image)
            # RGB -> BGR (for OpenCV)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
        elif isinstance(image, torch.Tensor):
            # Tensor -> numpy
            # 假设输入是 [C, H, W] 格式，范围 [0, 1] 或 [0, 255]
            if image.dim() == 3:
                image_np = image.permute(1, 2, 0).numpy()
                # 如果是归一化的，转回0-255
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
                # RGB -> BGR
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                # 不支持的格式，返回原图
                return inputs
                
        elif isinstance(image, np.ndarray):
            image_np = image.copy()
            # 假设已经是BGR格式
        else:
            # 不支持的类型，返回原图
            return inputs
        
        # 检测PCB边界
        bounds = self._detect_pcb_bounds(image_np)
        
        if bounds is None:
            # 检测失败
            if self.return_original_on_fail:
                return inputs
            else:
                raise ValueError("PCB detection failed")
        
        x1, y1, x2, y2 = bounds
        
        # 裁剪
        cropped = image_np[y1:y2, x1:x2]
        
        # 转回原始格式
        if isinstance(inputs[0], Image.Image):
            # BGR -> RGB -> PIL
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            result = Image.fromarray(cropped)
            
        elif isinstance(inputs[0], torch.Tensor):
            # BGR -> RGB -> Tensor
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            result = torch.from_numpy(cropped).permute(2, 0, 1)
            # 归一化到 [0, 1]
            if inputs[0].max() <= 1.0:
                result = result.float() / 255.0
                
        elif isinstance(inputs[0], np.ndarray):
            result = cropped
        else:
            result = inputs[0]
        
        # 返回所有输入，只修改第一个（图像）
        if len(inputs) == 1:
            return result
        else:
            return (result,) + inputs[1:]


class PCBBackgroundCropV2(v2.Transform):
    """PCB背景裁剪 - torchvision.transforms.v2 兼容版本.
    
    专门为torchvision v2 API设计的版本。
    
    Args:
        padding (int): 裁剪边距. Defaults to 10.
        min_area_ratio (float): PCB最小面积占比. Defaults to 0.1.
        
    Example:
        在ens_config.yaml中使用:
        ```yaml
        data:
          init_args:
            train_augmentations:
              - class_path: pcb_crop_transform.PCBBackgroundCropV2
                init_args:
                  padding: 10
                  min_area_ratio: 0.1
              - class_path: torchvision.transforms.v2.RandomRotation
                init_args:
                  degrees: 15
        ```
    """
    
    def __init__(self, padding: int = 10, min_area_ratio: float = 0.1):
        super().__init__()
        self.padding = padding
        self.min_area_ratio = min_area_ratio
    
    def _detect_and_crop(self, image: Image.Image) -> Image.Image:
        """检测并裁剪PCB."""
        # PIL -> numpy (RGB)
        image_np = np.array(image)
        
        # RGB -> BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        h, w = image_bgr.shape[:2]
        
        # 检测边界
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image  # 检测失败，返回原图
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < (h * w * self.min_area_ratio):
            return image  # 面积太小，返回原图
        
        x, y, w_pcb, h_pcb = cv2.boundingRect(largest)
        
        # 裁剪（加padding）
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(w, x + w_pcb + self.padding)
        y2 = min(h, y + h_pcb + self.padding)
        
        cropped_bgr = image_bgr[y1:y2, x1:x2]
        
        # BGR -> RGB -> PIL
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped_rgb)
    
    def forward(self, *inputs):
        """执行transform."""
        # 第一个输入应该是图像
        image = inputs[0]
        
        if isinstance(image, Image.Image):
            cropped = self._detect_and_crop(image)
        else:
            # 不是PIL Image，返回原样
            cropped = image
        
        # 返回处理后的图像和其他输入
        if len(inputs) == 1:
            return cropped
        else:
            return (cropped,) + inputs[1:]


# 确保可以被导入
__all__ = ["PCBBackgroundCrop", "PCBBackgroundCropV2"]

