# 1. Import required modules
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur
import cv2
import json

from anomalib.models import Patchcore
from visualize_heatmap import save_original_and_overlay_with_info


# ============================================================================
# ROI选择器类（整合自 roi_selector.py）
# ============================================================================

class ROISelector:
    """ROI选择工具类."""
    
    def __init__(self, image_path):
        """初始化ROI选择器.
        
        Args:
            image_path (str): 输入图像路径
        """
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(image_path))
        
        if self.image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        self.display_image = self.image.copy()
        self.rois = []  # 存储所有ROI [(x, y, w, h), ...]
        self.current_roi = None
        self.drawing = False
        self.start_point = None
        
        self.window_name = "ROI Selector - 's':Save  'r':Reset  'q':Quit"
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调函数."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.display_image = self.image.copy()
                self._draw_existing_rois()
                # 正在拖动的矩形框（粗线条，更醒目）
                cv2.rectangle(self.display_image, self.start_point, (x, y), (0, 255, 0), 5)
                cv2.imshow(self.window_name, self.display_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                
                # 计算ROI坐标(确保x1<x2, y1<y2)
                x1, x2 = min(self.start_point[0], end_point[0]), max(self.start_point[0], end_point[0])
                y1, y2 = min(self.start_point[1], end_point[1]), max(self.start_point[1], end_point[1])
                
                # 添加ROI（如果尺寸有效）
                if x2 - x1 > 5 and y2 - y1 > 5:
                    self.rois.append((x1, y1, x2 - x1, y2 - y1))
                    print(f"   ✓ 添加ROI #{len(self.rois)}: ({x1}, {y1}, {x2-x1}, {y2-y1})")
                
                self._update_display()
    
    def _draw_existing_rois(self):
        """在显示图像上绘制所有已存在的ROI."""
        for idx, (x, y, w, h) in enumerate(self.rois):
            # 已完成的ROI矩形框（粗线条）
            cv2.rectangle(self.display_image, (x, y), (x+w, y+h), (0, 255, 0), 4)
            # ROI标签文字
            cv2.putText(self.display_image, f"ROI {idx+1}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def _update_display(self, message=None, message_color=(0, 255, 0)):
        """更新显示图像.
        
        Args:
            message: 要显示的临时消息（可选）
            message_color: 消息颜色，默认绿色
        """
        self.display_image = self.image.copy()
        self._draw_existing_rois()
        
        # 添加提示信息背景
        h, w = self.image.shape[:2]
        info_bg = np.zeros((80, w, 3), dtype=np.uint8)
        cv2.putText(info_bg, f"ROI Count: {len(self.rois)}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_bg, "'s': Save  'r': Reset  'q': Quit", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # 如果有临时消息，在图像上叠加显示
        if message:
            # 在图像中央上方显示消息
            font_scale = 2.5  # 字体大小（增大）
            font_thickness = 4  # 字体粗细（增加）
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = 120  # 向下移动一点
            
            # 绘制半透明背景
            overlay = self.display_image.copy()
            padding = 30  # 增加边距
            cv2.rectangle(overlay, 
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, self.display_image, 0.3, 0, self.display_image)
            
            # 绘制消息文本
            cv2.putText(self.display_image, message, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, message_color, font_thickness)
        
        display = np.vstack([self.display_image, info_bg])
        cv2.imshow(self.window_name, display)
    
    def create_mask(self):
        """创建ROI mask（白色=ROI区域，黑色=忽略区域）."""
        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for x, y, roi_w, roi_h in self.rois:
            mask[y:y+roi_h, x:x+roi_w] = 255
        
        return mask
    
    def save_roi_config(self, output_path=None, save_dir=None):
        """保存ROI配置到JSON文件."""
        if not self.rois:
            print("   ⚠️  没有ROI可保存")
            return None
        
        # 确定保存目录（默认保存到 predict.py 所在目录）
        if save_dir is None:
            save_dir = Path(__file__).parent  # tools/tiled_ensemble/
        else:
            save_dir = Path(save_dir)
        
        if output_path is None:
            output_path = save_dir / "roi.json"
        
        config = {
            "image": str(self.image_path.name),
            "image_size": {
                "width": self.image.shape[1],
                "height": self.image.shape[0]
            },
            "rois": [
                {"x": x, "y": y, "width": w, "height": h}
                for x, y, w, h in self.rois
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # 同时保存mask图像
        mask = self.create_mask()
        mask_path = save_dir / "roi_mask.png"
        cv2.imwrite(str(mask_path), mask)
        
        print(f"   ✅ ROI配置已保存: {output_path}")
        print(f"   ✅ ROI mask已保存: {mask_path}")
        
        return str(output_path)
    
    def run(self, save_dir=None):
        """运行ROI选择器.
        
        Args:
            save_dir: ROI文件保存目录，默认为项目根目录
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self._update_display()
        
        saved_path = None
        message_timer = 0  # 消息显示计时器
        current_message = None
        message_color = (0, 255, 0)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # 更新消息显示
            if message_timer > 0:
                message_timer -= 1
                self._update_display(current_message, message_color)
                if message_timer == 0:
                    current_message = None
                    self._update_display()
            
            if key == ord('q'):
                # 退出：如果已保存则返回路径，否则返回 None
                break
            elif key == ord('r'):
                # 重置ROI
                self.rois = []
                print("   ✅ 已重置所有ROI")
                # 显示弹窗提示
                current_message = "ROI Reset!"
                message_color = (0, 165, 255)  # 橙色
                message_timer = 100  # 显示约1秒（100帧 @ 1ms/frame）
                self._update_display(current_message, message_color)
            elif key == ord('s'):
                # 保存ROI，但继续编辑
                saved_path = self.save_roi_config(save_dir=save_dir)
                if saved_path:
                    print("   💡 继续编辑或按 'q' 退出")
                    # 显示弹窗提示
                    current_message = "ROI Saved"
                    message_color = (0, 255, 0)  # 绿色
                    message_timer = 100  # 显示约1秒
                    self._update_display(current_message, message_color)
                else:
                    # 保存失败
                    current_message = "No ROI to Save"
                    message_color = (0, 0, 255)  # 红色
                    message_timer = 100
                    self._update_display(current_message, message_color)
        
        cv2.destroyAllWindows()
        return saved_path


def load_roi_mask(roi_config_path):
    """从ROI配置文件加载mask."""
    roi_config_path = Path(roi_config_path)
    
    if not roi_config_path.exists():
        raise FileNotFoundError(f"ROI配置文件不存在: {roi_config_path}")
    
    with open(roi_config_path, 'r') as f:
        config = json.load(f)
    
    # 创建mask
    h = config["image_size"]["height"]
    w = config["image_size"]["width"]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for roi in config["rois"]:
        x, y = roi["x"], roi["y"]
        roi_w, roi_h = roi["width"], roi["height"]
        mask[y:y+roi_h, x:x+roi_w] = 255
    
    return mask


# ============================================================================
# PCB背景裁剪功能（与 pcb_crop_transform.py 相同的算法）
# ============================================================================

def detect_and_crop_pcb(image: Image.Image, padding: int = 10, min_area_ratio: float = 0.1) -> Image.Image | None:
    """检测并裁剪PCB背景.
    
    Args:
        image (Image.Image): 输入PIL图像
        padding (int): 裁剪边距（像素）. Defaults to 10.
        min_area_ratio (float): PCB最小面积占比. Defaults to 0.1.
        
    Returns:
        Image.Image | None: 裁剪后的图像，失败返回 None
    """
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
        return None
    
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    if area < (h * w * min_area_ratio):
        return None
    
    x, y, w_pcb, h_pcb = cv2.boundingRect(largest)
    
    # 裁剪（加padding）
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + w_pcb + padding)
    y2 = min(h, y + h_pcb + padding)
    
    cropped_bgr = image_bgr[y1:y2, x1:x2]
    
    # BGR -> RGB -> PIL
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)


# ============================================================================
# 加载配置文件
# ============================================================================

import yaml

# 配置文件路径
CONFIG_FILE = Path(__file__).parent / "predict_config.yaml"

if not CONFIG_FILE.exists():
    print(f"❌ 配置文件不存在: {CONFIG_FILE}")
    print(f"请确保 predict_config.yaml 文件在 tools/tiled_ensemble/ 目录下")
    exit(1)

print("="*70)
print("🚀 Tiled Ensemble 集成预测")
print("="*70)
print(f"📝 加载配置文件: {CONFIG_FILE.name}\n")

with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 提取配置参数
CHECKPOINT_DIR = Path(config['paths']['checkpoint_dir'])
INPUT_DIR = Path(config['paths']['input_dir'])
OUTPUT_DIR = Path(config['paths']['output_dir'])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = config['image']['image_size']
TILE_SIZE = config['image']['tile_size']

APPLY_SEAM_SMOOTHING = config['seam_smoothing']['apply']
SEAM_SIGMA = config['seam_smoothing']['sigma']
SEAM_WIDTH = config['seam_smoothing']['width']

USE_ROI = config['roi']['enable']
ROI_CONFIG_PATH = config['roi']['config_path']

ENABLE_PCB_CROP = config['pcb_crop']['enable']
PCB_CROP_PADDING = config['pcb_crop']['padding']
PCB_MIN_AREA_RATIO = config['pcb_crop']['min_area_ratio']

NORMALIZED_THRESHOLD = config['threshold']['normalized_threshold']

print("⚙️  配置信息:")
print(f"   - 输入目录: {INPUT_DIR}")
print(f"   - 输出目录: {OUTPUT_DIR}")
print(f"   - 模型目录: {CHECKPOINT_DIR}")
print(f"   - 判断阈值: {NORMALIZED_THRESHOLD}")
print(f"   - PCB裁剪: {'启用' if ENABLE_PCB_CROP else '禁用'}")
print(f"   - ROI过滤: {'启用' if USE_ROI else '禁用'}")
print(f"   - 接缝平滑: {'启用' if APPLY_SEAM_SMOOTHING else '禁用'}")
print("="*70)
print("工作原理：")
print("  0. 自动裁剪PCB背景")
print("  1. 图像调整为 512×512")
print("  2. 切分成 2×2 = 4 个 tiles（每个256×256）")
print("  3. 每个tile由对应的模型预测：")
print("     - model0_0.ckpt → Tile 0 (左上)")
print("     - model0_1.ckpt → Tile 1 (右上)")
print("     - model1_0.ckpt → Tile 2 (左下)")
print("     - model1_1.ckpt → Tile 3 (右下)")
print("  4. 合并所有tile的预测结果")
print("="*70)

# 3. 加载训练时的统计信息（用于归一化）
print("\n[步骤 1/5] 📊 加载归一化统计信息...")
import json
stats_file = CHECKPOINT_DIR / "stats.json"
if not stats_file.exists():
    print(f"   ❌ 找不到 stats.json: {stats_file}")
    print(f"   这个文件包含了训练时的归一化参数，是必需的！")
    exit(1)

with open(stats_file, 'r') as f:
    stats = json.load(f)

# 获取归一化参数
ANOMALY_MAP_MIN = stats["minmax"]["anomaly_map"]["min"]
ANOMALY_MAP_MAX = stats["minmax"]["anomaly_map"]["max"]
PIXEL_THRESHOLD = stats["pixel_threshold"]  # 用于anomaly_map归一化
IMAGE_THRESHOLD = stats["image_threshold"]  # 用于pred_score归一化

PRED_SCORE_MIN = stats["minmax"]["pred_score"]["min"]
PRED_SCORE_MAX = stats["minmax"]["pred_score"]["max"]

print(f"   ✅ Anomaly Map 范围: [{ANOMALY_MAP_MIN:.4f}, {ANOMALY_MAP_MAX:.4f}]")
print(f"   ✅ Pixel 阈值: {PIXEL_THRESHOLD:.4f} (归一化后 = 0.5)")
print(f"   ✅ Pred Score 范围: [{PRED_SCORE_MIN:.4f}, {PRED_SCORE_MAX:.4f}]")
print(f"   ✅ Image 阈值: {IMAGE_THRESHOLD:.4f} (归一化后 = 0.5)")

# 4. 加载4个模型
print("\n[步骤 2/5] 🔍 加载4个模型...")
models = {}
model_files = [
    ("model0_0.ckpt", (0, 0)),  # 左上
    ("model0_1.ckpt", (0, 1)),  # 右上
    ("model1_0.ckpt", (1, 0)),  # 左下
    ("model1_1.ckpt", (1, 1)),  # 右下
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   使用设备: {device}")

for model_file, position in model_files:
    model_path = CHECKPOINT_DIR / model_file
    if not model_path.exists():
        print(f"   ❌ 找不到模型: {model_path}")
        print(f"   请检查路径是否正确")
        exit(1)
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # 禁用预处理器和后处理器，因为我们手动处理
    model = Patchcore(pre_processor=False, post_processor=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)  # strict=False忽略post_processor参数
    model.to(device)
    model.eval()  # 确保是评估模式
    
    models[position] = model
    print(f"   ✅ {model_file} → Tile 位置 {position}")

print(f"   成功加载 {len(models)} 个模型")


# 5. 图像预处理
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def normalize_anomaly_map(anomaly_map, threshold, min_val, max_val):
    """
    将anomaly map归一化到0-1范围
    
    使用与训练pipeline相同的归一化方法：
    将threshold映射到0.5，其他值按比例缩放
    """
    # 使用Anomalib的归一化公式
    normalized = ((anomaly_map - threshold) / (max_val - min_val)) + 0.5
    # 限制在0-1范围内
    normalized = torch.clamp(normalized, 0.0, 1.0)
    return normalized


def split_into_tiles(image_tensor):
    """将 512×512 图像切分成 2×2 个 256×256 tiles"""
    tiles = {}
    for i in range(2):  # 行
        for j in range(2):  # 列
            # 提取tile
            tile = image_tensor[
                :,
                i * TILE_SIZE : (i + 1) * TILE_SIZE,
                j * TILE_SIZE : (j + 1) * TILE_SIZE,
            ]
            tiles[(i, j)] = tile
    return tiles


def merge_tiles(tile_predictions):
    """
    合并4个tile的预测结果（不归一化，保持原始值）
    
    Args:
        tile_predictions: 字典，{(i,j): {'anomaly_map': tensor, 'pred_score': float}}
    
    Returns:
        merged_anomaly_map: 合并后的anomaly map
        merged_pred_score: 所有tiles的平均pred_score
    """
    # 创建完整的anomaly map
    full_map = torch.zeros((IMAGE_SIZE, IMAGE_SIZE), device=device)
    
    # 收集所有tile的pred_score（用于计算平均值）
    tile_scores = []
    
    for (i, j), pred_data in tile_predictions.items():
        pred_map = pred_data['anomaly_map']
        pred_score = pred_data['pred_score']
        
        # 确保pred_map是2D的
        if pred_map.dim() > 2:
            pred_map = pred_map.squeeze()
        
        # 调整大小到256×256（如果需要）
        if pred_map.shape != (TILE_SIZE, TILE_SIZE):
            pred_map = F.interpolate(
                pred_map.unsqueeze(0).unsqueeze(0),
                size=(TILE_SIZE, TILE_SIZE),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # 放到对应位置（不归一化，保持原始值！）
        full_map[
            i * TILE_SIZE : (i + 1) * TILE_SIZE,
            j * TILE_SIZE : (j + 1) * TILE_SIZE,
        ] = pred_map
        
        # 收集pred_score
        tile_scores.append(pred_score)
    
    # pred_score是所有tiles的平均值（与训练时一致）
    avg_pred_score = sum(tile_scores) / len(tile_scores)
    
    return full_map, avg_pred_score


def smooth_seams(anomaly_map, sigma=2, width_ratio=0.1):
    """对 tile 接缝区域做高斯平滑，减轻硬拼接痕迹。"""
    # anomaly_map: [H, W]，此处 H=W=IMAGE_SIZE
    h, w = anomaly_map.shape
    assert h == IMAGE_SIZE and w == IMAGE_SIZE, "anomaly_map 尺寸必须等于 IMAGE_SIZE"

    # 计算接缝区域宽度（至少 1 像素）
    seam_w = max(1, int(width_ratio * TILE_SIZE))

    # 构造接缝 mask（只在接缝附近进行平滑融合）
    mask = torch.zeros_like(anomaly_map)
    # 横向接缝（行方向）：位于 TILE_SIZE 行附近
    row_start = max(0, TILE_SIZE - seam_w)
    row_end = min(IMAGE_SIZE, TILE_SIZE + seam_w)
    mask[row_start:row_end, :] = 1.0
    # 纵向接缝（列方向）：位于 TILE_SIZE 列附近
    col_start = max(0, TILE_SIZE - seam_w)
    col_end = min(IMAGE_SIZE, TILE_SIZE + seam_w)
    mask[:, col_start:col_end] = 1.0

    # 生成合适的核大小（奇数）
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)

    # 对整幅图做高斯模糊，再用 mask 只替换接缝区域
    blurred = tv_gaussian_blur(anomaly_map.unsqueeze(0), kernel_size=ksize, sigma=sigma).squeeze(0)
    smoothed = anomaly_map * (1.0 - mask) + blurred * mask
    return smoothed


# 6. ROI 选择（如果启用且未指定ROI文件）
if USE_ROI and ROI_CONFIG_PATH is None:
    print("\n[步骤 3/6] 🎯 选择 ROI 区域...")
    print("="*70)
    
    # 获取预测目录中的图像列表
    temp_image_files = list(INPUT_DIR.glob("*.jpg")) + \
                      list(INPUT_DIR.glob("*.jpeg")) + \
                      list(INPUT_DIR.glob("*.png"))
    
    if len(temp_image_files) == 0:
        print(f"   ❌ 在 {INPUT_DIR} 中没有找到图像文件")
        print(f"   请将要预测的图片放到该文件夹中")
        exit(1)
    
    # 使用第一张图片来选择ROI（或让用户选择）
    print(f"   找到 {len(temp_image_files)} 张图像")
    print(f"\n   准备参考图片用于选择ROI区域...")
    print(f"   参考图片: {temp_image_files[0].name}")
    
    # 🔑 关键修改：先裁剪 PCB，然后在裁剪后的图片上选择 ROI
    try:
        reference_image = Image.open(temp_image_files[0]).convert("RGB")
        print(f"   原始尺寸: {reference_image.size[0]}×{reference_image.size[1]}")
        
        # PCB 裁剪
        if ENABLE_PCB_CROP:
            cropped_ref = detect_and_crop_pcb(reference_image, padding=PCB_CROP_PADDING, min_area_ratio=PCB_MIN_AREA_RATIO)
            if cropped_ref is not None:
                print(f"   ✂️  PCB裁剪: {reference_image.size[0]}×{reference_image.size[1]} → {cropped_ref.size[0]}×{cropped_ref.size[1]}")
                reference_image = cropped_ref
            else:
                print(f"   ⚠️  PCB裁剪失败，使用原图")
        
        # 保存临时裁剪图用于 ROI 选择
        temp_cropped_path = Path(__file__).parent / "temp_cropped_for_roi.png"
        reference_image.save(temp_cropped_path)
        print(f"   ✅ 裁剪后图片已保存（用于ROI选择）")
        
        print(f"\n   操作说明:")
        print(f"     - 左键拖动: 选择矩形ROI区域")
        print(f"     - 's': 保存ROI配置（可多次保存）")
        print(f"     - 'r': 重置所有ROI")
        print(f"     - 'q': 确认退出并继续预测")
        print("="*70)
        
        # predict.py 所在目录（保存ROI文件的位置）
        script_dir = Path(__file__).parent
        
        # 使用裁剪后的图片进行 ROI 选择
        selector = ROISelector(str(temp_cropped_path))
        saved_roi_path = selector.run(save_dir=script_dir)
        
        # 删除临时文件
        if temp_cropped_path.exists():
            temp_cropped_path.unlink()
            print(f"   🧹 临时文件已清理")
        
        
        if saved_roi_path:
            ROI_CONFIG_PATH = saved_roi_path
            print(f"\n   ✅ ROI配置已保存（基于裁剪后图片），将应用于所有图像")
            print(f"   📁 文件: {Path(saved_roi_path).name}")
        else:
            print(f"\n   ⚠️  未保存ROI配置，将不使用ROI过滤")
            USE_ROI = False
    except Exception as e:
        print(f"\n   ❌ ROI选择失败: {e}")
        print(f"   将不使用ROI过滤")
        USE_ROI = False
        import traceback
        traceback.print_exc()

# 7. 遍历所有图像进行预测
print("\n[步骤 4/6] 📂 加载图像...")
image_files = list(INPUT_DIR.glob("*.jpg")) + \
              list(INPUT_DIR.glob("*.jpeg")) + \
              list(INPUT_DIR.glob("*.png"))

if len(image_files) == 0:
    print(f"   ❌ 在 {INPUT_DIR} 中没有找到图像文件")
    print(f"   请将要预测的图片放到该文件夹中")
    exit(1)

print(f"   找到 {len(image_files)} 张图像")


print("\n[步骤 5/6] 🔮 使用4个模型进行集成预测...")
print("="*70)

for img_idx, image_path in enumerate(image_files, 1):
    print(f"\n[{img_idx}/{len(image_files)}] 处理: {image_path.name}")
    
    # 读取图像
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    print(f"   原始尺寸: {original_size[0]}×{original_size[1]}")
    
    # PCB背景裁剪（如果启用）
    if ENABLE_PCB_CROP:
        cropped = detect_and_crop_pcb(image, padding=PCB_CROP_PADDING, min_area_ratio=PCB_MIN_AREA_RATIO)
        if cropped is not None:
            print(f"   ✂️  PCB裁剪: {image.size[0]}×{image.size[1]} → {cropped.size[0]}×{cropped.size[1]}")
            image = cropped
            original_size = image.size  # 更新原始尺寸为裁剪后的尺寸
        else:
            print(f"   ⚠️  PCB裁剪失败，使用原图")
    
    # 加载ROI mask（如果启用）
    roi_mask = None
    if USE_ROI and ROI_CONFIG_PATH:
        # 使用指定的ROI配置文件（批量应用相同ROI）
        try:
            roi_mask = load_roi_mask(ROI_CONFIG_PATH)
            if img_idx == 1:  # 只在第一张图片时打印
                print(f"   ✅ ROI配置: {Path(ROI_CONFIG_PATH).name}")
        except Exception as e:
            print(f"   ⚠️  加载ROI mask失败: {e}")
            roi_mask = None
    
    # 预处理：调整为512×512并归一化
    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 512, 512]
    print(f"   调整为 512×512")
    
    # ✅ 关键修正：先切分图像，然后每个模型预测对应的tile
    # 步骤1: 将512×512图像切分成4个256×256的tiles
    tiles = split_into_tiles(image_tensor.squeeze(0))  # 移除batch维度
    print(f"   切分成 {len(tiles)} 个tiles (每个256×256)")
    
    # 步骤2: 每个模型预测对应位置的tile
    tile_predictions = {}
    with torch.no_grad():
        for position, model in models.items():
            # 取出对应位置的tile（256×256）
            tile = tiles[position].unsqueeze(0)  # [1, 3, 256, 256]
            
            # 用这个tile预测
            output = model(tile)
            anomaly_map = output.anomaly_map.squeeze()  # 获取anomaly map  
            pred_score = output.pred_score.item()  # 获取pred_score
            
            tile_predictions[position] = {
                'anomaly_map': anomaly_map,
                'pred_score': pred_score
            }
            print(f"   ✓ Tile {position} 预测完成 (原始分数: {pred_score:.2f})")
    
    # 合并所有tile的预测（保持原始值）
    print(f"   🔄 合并4个tile的预测结果...")
    merged_anomaly_map, merged_pred_score = merge_tiles(tile_predictions)

    # 可选：对接缝区域做平滑（与官方 pipeline 的 SeamSmoothing 对齐）
    if APPLY_SEAM_SMOOTHING:
        merged_anomaly_map = smooth_seams(
            merged_anomaly_map,
            sigma=SEAM_SIGMA,
            width_ratio=SEAM_WIDTH,
        )
    
    print(f"   📊 合并后的原始值:")
    print(f"      - Pred Score (tiles平均): {merged_pred_score:.2f}")
    print(f"      - Anomaly Map 最大值: {merged_anomaly_map.max().item():.2f}")
    
    # 归一化 (IMAGE级别，与训练时一致)
    print(f"   🎯 归一化到 [0, 1] 范围...")
    
    # 归一化 pred_score
    normalized_pred_score = normalize_anomaly_map(
        torch.tensor(merged_pred_score, device=device),
        IMAGE_THRESHOLD,    # threshold
        PRED_SCORE_MIN,     # min_val
        PRED_SCORE_MAX      # max_val
    ).item()
    
    # 归一化 anomaly_map
    normalized_anomaly_map = normalize_anomaly_map(
        merged_anomaly_map,
        PIXEL_THRESHOLD,    # threshold
        ANOMALY_MAP_MIN,    # min_val
        ANOMALY_MAP_MAX     # max_val
    )
    
    # 应用ROI mask（如果启用）
    if roi_mask is not None:
        # 将ROI mask调整到512×512
        roi_mask_resized = cv2.resize(roi_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        roi_mask_tensor = torch.from_numpy(roi_mask_resized / 255.0).float().to(device)
        
        # 应用mask：ROI外的区域anomaly值设为0
        normalized_anomaly_map = normalized_anomaly_map * roi_mask_tensor
        
        # 重新计算pred_score（只基于ROI内的区域）
        # 这样pred_score只反映ROI内的异常程度，不受ROI外区域影响
        roi_area = roi_mask_tensor.sum().item()
        if roi_area > 0:
            # ROI内的平均anomaly值
            roi_anomaly_sum = (normalized_anomaly_map * roi_mask_tensor).sum().item()
            normalized_pred_score = roi_anomaly_sum / roi_area
        else:
            # 如果ROI为空，则pred_score设为0
            normalized_pred_score = 0.0
        
        print(f"   🎯 已应用ROI mask，pred_score已重新计算（仅基于ROI内区域）")
    
    print(f"   📊 归一化后:")
    print(f"      - Pred Score: {normalized_pred_score:.4f}")
    print(f"      - Anomaly Map 最大值: {normalized_anomaly_map.max().item():.4f}")
    
    # 判断策略：基于整体分数
    # ============================================================
    # pred_score (整体分数): 反映整张图的平均异常程度
    # 阈值从配置文件读取: NORMALIZED_THRESHOLD
    # ============================================================
    
    # 基于整体分数判断
    if normalized_pred_score >= NORMALIZED_THRESHOLD:
        pred_label = "NG"
        reason = f"异常（分数 {normalized_pred_score:.3f} ≥ {NORMALIZED_THRESHOLD}）"
    else:
        pred_label = "OK"
        reason = f"正常（分数 {normalized_pred_score:.3f} < {NORMALIZED_THRESHOLD}）"
    
    print(f"   ✅ 判断结果: {pred_label} - {reason}")
    
    # 用于可视化的分数（使用归一化后的pred_score）
    pred_score = normalized_pred_score
    
    # 调整归一化后的anomaly map尺寸到原始图像大小
    anomaly_map_resized = F.interpolate(
        normalized_anomaly_map.unsqueeze(0).unsqueeze(0),
        size=original_size[::-1],  # (height, width)
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    # 保存可视化结果
    print(f"   💾 保存可视化结果...")
    save_original_and_overlay_with_info(
        str(image_path),
        anomaly_map_resized,
        pred_label,
        pred_score,
        save_dir=str(OUTPUT_DIR),
        image_obj=image,  # 传递裁剪后的图像对象
    )

print("\n" + "="*70)
print("[步骤 6/6] ✅ 所有预测完成！")
print(f"📁 结果保存在: {OUTPUT_DIR}")
print("="*70)
print(f"\n💡 Tiled Ensemble 预测流程:")
print(f"   1️⃣  Predict: 每个tile独立预测（原始值）")
print(f"   2️⃣  Merge: 合并tiles的anomaly_map和pred_score(平均)")
print(f"   3️⃣  Normalize: 在IMAGE级别归一化（threshold→0.5）")
print(f"   4️⃣  Threshold: 使用pred_score判断（阈值={NORMALIZED_THRESHOLD:.2f}）")
print(f"\n📊 参数:")
print(f"   - Pred Score 范围: [{PRED_SCORE_MIN:.2f}, {PRED_SCORE_MAX:.2f}]")
print(f"   - Image Threshold: {IMAGE_THRESHOLD:.2f} → 归一化后 = 0.50")
print(f"   - Anomaly Map 范围: [{ANOMALY_MAP_MIN:.2f}, {ANOMALY_MAP_MAX:.2f}]")
print(f"   - Pixel Threshold: {PIXEL_THRESHOLD:.2f} → 归一化后 = 0.50")
print(f"\n🎯 判断规则:")
print(f"   - 分数 ≥ {NORMALIZED_THRESHOLD:.2f} → NG (异常)")
print(f"   - 分数 < {NORMALIZED_THRESHOLD:.2f} → OK (正常)")
print(f"\n⚙️  配置文件: {CONFIG_FILE.name}")
print("="*70)