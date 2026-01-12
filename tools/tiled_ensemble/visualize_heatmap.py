import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def save_original_and_overlay_with_info(
    image_path,
    anomaly_map,
    pred_label,
    pred_score,
    save_dir,
    image_obj=None,
):
    """保存原图、叠加图和异常区域的可视化结果.
    
    Args:
        image_path: 图像路径（用于生成保存文件名）
        anomaly_map: 异常热力图
        pred_label: 预测标签（"OK" 或 "NG"）
        pred_score: 预测分数
        save_dir: 保存目录
        image_obj: PIL Image对象（可选）。如果提供，使用此对象而不是从路径读取
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
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. anomaly map
    heatmap = anomaly_map.squeeze().cpu().numpy()
    heatmap_norm = cv2.normalize(
        heatmap, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

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
    
    # 在异常区域上叠加热力图
    masked_overlay = np.zeros_like(image)
    masked_overlay[anomaly_mask_3ch > 0] = overlay[anomaly_mask_3ch > 0]

    # ===== 文本信息 =====
    label_text = pred_label          # 已经是 "NG" / "OK"
    score_value = float(pred_score)  # 已经是 float
    text = f"{label_text} | score = {score_value:.3f}"
    text_color = "red" if label_text == "NG" else "lime"

    # 5. 画整张结果图（3张图）
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Anomaly Overlay")
    axes[1].axis("off")

    axes[2].imshow(masked_image)
    axes[2].set_title("Anomaly Region (Original)")
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
