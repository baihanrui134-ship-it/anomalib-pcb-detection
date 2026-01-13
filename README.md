# Tiled Ensemble 异常检测工具

基于 Anomalib 的分块集成异常检测工具，专为 PCB 等大尺寸图像设计。

## 📋 目录

- [功能特性](#功能特性)
- [文件结构](#文件结构)
- [快速开始](#快速开始)
  - [训练模型](#训练模型)
  - [预测推理](#预测推理)
- [配置说明](#配置说明)
- [进阶功能](#进阶功能)

---

## 🎯 功能特性

- **分块集成训练**：将大尺寸图像切分成多个 tiles，每个 tile 训练独立模型
- **智能接缝平滑**：自动平滑相邻 tiles 拼接处，消除接缝伪影
- **PCB 背景裁剪**：自动检测并裁剪 PCB 边缘，去除黑色背景干扰
- **ROI 区域选择**：支持指定感兴趣区域，忽略不相关区域的异常
- **可视化输出**：生成原图、热力图叠加、异常区域三合一可视化结果

---

## 📁 文件结构

```
tools/tiled_ensemble/
├── train.py                    # 训练脚本
├── predict.py                  # 预测脚本（独立版，包含完整功能）
├── eval.py                     # 评估脚本
├── ens_config.yaml             # 训练配置文件
├── predict_config.yaml         # 预测配置文件
├── visualize_heatmap.py        # 热图可视化工具
├── roi.json                    # ROI 配置文件（预测时生成）
├── roi_mask.png                # ROI 掩码图像（预测时生成）
└── README.md                   # 本文档
```

---

## 🚀 快速开始

### 训练模型

#### 1. 准备数据集

按照以下结构组织您的数据：

```
datasets/pcb/
├── good/              # 正常样本（训练 + 测试）
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── defect/            # 异常样本（仅测试）
    ├── defect_001.jpg
    ├── defect_002.jpg
    └── ...
```

#### 2. 配置训练参数

编辑 `ens_config.yaml`，根据需要调整参数：

```yaml
# 数据配置
data:
  class_path: anomalib.data.Folder
  init_args:
    root: datasets/pcb          # 数据集根目录
    normal_dir: good            # 正常样本目录
    abnormal_dir: defect        # 异常样本目录
    normal_split_ratio: 0.2     # 20% 正常样本用于测试

# 分块配置
tiling:
  image_size: [512, 512]        # 调整后的图像尺寸
  tile_size: [256, 256]         # 每个 tile 的尺寸
  stride: 256                   # 步长（256 表示无重叠）

# 模型配置
TrainModels:
  model:
    class_path: Patchcore       # 使用 PatchCore 模型
  trainer:
    max_epochs: 1               # PatchCore 只需 1 个 epoch
```

**关键参数说明：**

- `image_size: [512, 512]` + `tile_size: [256, 256]` + `stride: 256`
  - 图像被切分成 **2×2 = 4 个 tiles**
  - 每个 tile 训练一个独立的 PatchCore 模型
  
- `normalization_stage: image`
  - 在整张图像级别进行归一化（推荐）
  
- `SeamSmoothing`
  - 自动平滑 tile 接缝区域，消除拼接痕迹

#### 3. 运行训练

```bash
cd tools/tiled_ensemble
python train.py
```

**训练流程：**

1. 加载数据集并应用 PCB 背景裁剪（如果启用）
2. 将图像调整为 512×512
3. 切分成 4 个 256×256 的 tiles
4. 为每个 tile 训练独立的 PatchCore 模型
5. 保存 4 个模型权重文件：
   - `model0_0.ckpt` - 左上 tile
   - `model0_1.ckpt` - 右上 tile
   - `model1_0.ckpt` - 左下 tile
   - `model1_1.ckpt` - 右下 tile
6. 保存归一化统计信息：`stats.json`
7. 自动运行测试评估

**训练结果保存位置：**

```
results/PatchCore/pcb/latest/
├── weights/
│   └── lightning/
│       ├── model0_0.ckpt       # 左上 tile 模型
│       ├── model0_1.ckpt       # 右上 tile 模型
│       ├── model1_0.ckpt       # 左下 tile 模型
│       ├── model1_1.ckpt       # 右下 tile 模型
│       └── stats.json          # 归一化统计信息
└── images/                     # 测试结果可视化
```

---

### 预测推理

#### 1. 准备预测数据

将待预测的图像放入指定目录：

```
datasets/pcb/predict/
├── test_001.jpg
├── test_002.png
└── ...
```

#### 2. 配置预测参数

编辑 `predict_config.yaml`：

```yaml
# 路径配置
paths:
  checkpoint_dir: "results/PatchCore/pcb/latest/weights/lightning"  # 模型目录
  input_dir: "datasets/pcb/predict"                                 # 输入目录
  output_dir: "datasets/pcb/predict_results"                        # 输出目录

# PCB 裁剪配置
pcb_crop:
  enable: true              # 启用 PCB 背景裁剪
  padding: 10               # 裁剪边距（像素）
  min_area_ratio: 0.1       # PCB 最小面积占比

# 接缝平滑配置
seam_smoothing:
  apply: true               # 启用接缝平滑
  sigma: 2                  # 高斯核 sigma
  width: 0.1                # 平滑区域宽度占比

# ROI 配置
roi:
  enable: true              # 启用 ROI 过滤
  config_path: null         # null = 首次运行时选择 ROI

# 判断阈值
threshold:
  normalized_threshold: 0.5 # 判断阈值（标准值 0.5）
                            # 分数 ≥ 0.5 → NG (异常)
                            # 分数 < 0.5 → OK (正常)
```

#### 3. 运行预测

```bash
cd tools/tiled_ensemble
python predict.py
```

**预测流程：**

```
步骤 1: 加载归一化统计信息 (stats.json)
    ↓
步骤 2: 加载 4 个模型 (model0_0.ckpt ~ model1_1.ckpt)
    ↓
步骤 3: [可选] ROI 选择
    - 如果 enable: true 且 config_path: null
    - 会弹出窗口让您选择 ROI 区域
    - 操作：
      * 左键拖动：选择矩形 ROI
      * 's': 保存 ROI 配置
      * 'r': 重置所有 ROI
      * 'q': 完成并继续
    ↓
步骤 4: 遍历所有图像
    ├── PCB 背景裁剪（自动去除黑边）
    ├── 调整为 512×512
    ├── 切分成 4 个 tiles (256×256)
    ├── 每个 tile 用对应模型预测
    ├── 合并预测结果
    ├── 接缝平滑（消除拼接痕迹）
    ├── 归一化到 [0, 1]
    ├── [可选] 应用 ROI mask
    └── 判断 OK/NG
    ↓
步骤 5: 保存可视化结果
```

**预测结果输出：**

```
datasets/pcb/predict_results/
├── test_001_OK_0.234.jpg       # 正常样本（分数 < 0.5）
├── test_002_NG_0.678.jpg       # 异常样本（分数 ≥ 0.5）
└── ...
```

每张结果图包含三个子图：
1. **Original** - 原始图像（裁剪后）
2. **Anomaly Overlay** - 热力图叠加（显示判断结果和分数）
3. **Anomaly Region** - 异常区域高亮（黑色背景 + 红色边框）

---

## ⚙️ 配置说明

### 训练配置 (ens_config.yaml)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `tiling.image_size` | 调整后的图像尺寸 | [512, 512] |
| `tiling.tile_size` | 每个 tile 的尺寸 | [256, 256] |
| `tiling.stride` | 切分步长 | 256 |
| `normalization_stage` | 归一化阶段 | image |
| `thresholding_stage` | 阈值应用阶段 | image |
| `SeamSmoothing.apply` | 是否平滑接缝 | True |
| `SeamSmoothing.sigma` | 高斯核 sigma | 2 |
| `SeamSmoothing.width` | 平滑区域宽度因子 | 0.1 |
| `data.normal_split_ratio` | 正常样本测试集比例 | 0.2 |
| `TrainModels.trainer.max_epochs` | 训练轮数 | 1 |

### 预测配置 (predict_config.yaml)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `paths.checkpoint_dir` | 模型权重目录 | results/PatchCore/pcb/latest/weights/lightning |
| `paths.input_dir` | 输入图像目录 | datasets/pcb/predict |
| `paths.output_dir` | 结果输出目录 | datasets/pcb/predict_results |
| `pcb_crop.enable` | 启用 PCB 裁剪 | true |
| `pcb_crop.padding` | 裁剪边距（像素） | 10 |
| `pcb_crop.min_area_ratio` | PCB 最小面积占比 | 0.1 |
| `seam_smoothing.apply` | 启用接缝平滑 | true |
| `seam_smoothing.sigma` | 高斯核 sigma | 2 |
| `seam_smoothing.width` | 平滑区域宽度占比 | 0.1 |
| `roi.enable` | 启用 ROI 过滤 | true |
| `roi.config_path` | ROI 配置文件路径 | null |
| `threshold.normalized_threshold` | 判断阈值 | 0.5 |

---

## 🔧 进阶功能

### 1. PCB 背景裁剪

**功能：** 自动检测 PCB 边缘，裁剪黑色背景，保留有效区域。

**原理：**
- Canny 边缘检测
- 形态学膨胀和闭运算
- 寻找最大轮廓
- 计算包围矩形并裁剪

**配置：**

```yaml
pcb_crop:
  enable: true              # 启用裁剪
  padding: 10               # 边距（像素）
  min_area_ratio: 0.1       # 最小面积占比（过滤误检）
```

**效果：**
- 输入：2000×2000 图像（大部分是黑色背景）
- 输出：1200×1400 图像（仅 PCB 区域 + 10px 边距）

### 2. ROI 区域选择

**功能：** 指定感兴趣区域，忽略其他区域的异常。

**使用场景：**
- PCB 上有固定的文字区域（正常但可能被误判为异常）
- 只关心特定区域的缺陷
- 边缘区域容易误报

**操作步骤：**

1. 配置启用 ROI：
   ```yaml
   roi:
     enable: true
     config_path: null  # 首次运行时选择
   ```

2. 运行预测：
   ```bash
   python predict.py
   ```

3. 在弹出的窗口中：
   - **左键拖动**：框选 ROI 区域（可以选多个）
   - **'s' 键**：保存 ROI 配置（保存到 `roi.json` 和 `roi_mask.png`）
   - **'r' 键**：重置所有 ROI
   - **'q' 键**：完成并继续预测

4. 后续预测会自动使用保存的 ROI：
   ```yaml
   roi:
     enable: true
     config_path: "roi.json"  # 自动加载
   ```

**工作原理：**
- 预测时只计算 ROI 内的异常分数
- ROI 外的区域异常值被置为 0
- 最终分数 = ROI 内平均异常值

### 3. 接缝平滑

**功能：** 平滑相邻 tiles 拼接处，消除接缝伪影。

**原理：**
- 识别 tile 接缝位置（行/列边界）
- 构建接缝 mask（接缝周围一定宽度）
- 对接缝区域应用高斯模糊
- 融合平滑后的结果

**配置：**

```yaml
seam_smoothing:
  apply: true               # 启用平滑
  sigma: 2                  # 高斯核 sigma（越大越模糊）
  width: 0.1                # 平滑区域宽度占比（相对 tile 边长）
```

**效果：**
- 消除 tile 拼接处的明显边界
- 使热力图更加连续平滑

### 4. 判断阈值调整

**功能：** 调整 OK/NG 判断的敏感度。

**配置：**

```yaml
threshold:
  normalized_threshold: 0.5  # 默认值
```

**调整建议：**
- **提高阈值（如 0.6）**：减少误报（NG → OK），可能漏检
- **降低阈值（如 0.4）**：减少漏检（OK → NG），可能误报
- **建议范围**：0.4 ~ 0.6

**示例：**

| 阈值 | 效果 | 适用场景 |
|------|------|----------|
| 0.3 | 极其敏感 | 允许误报，绝不漏检（如医疗） |
| 0.4 | 较敏感 | 减少漏检 |
| 0.5 | 标准 | 平衡误报和漏检（推荐） |
| 0.6 | 较宽松 | 减少误报 |
| 0.7 | 宽松 | 只标记明显异常 |

### 5. 归一化说明

**训练时：**
1. 收集所有训练样本的异常分数
2. 计算阈值（如 F1-score 最优点）
3. 保存统计信息到 `stats.json`：
   ```json
   {
     "minmax": {
       "anomaly_map": {"min": 0.0, "max": 25.5},
       "pred_score": {"min": 2.1, "max": 18.3}
     },
     "pixel_threshold": 12.4,
     "image_threshold": 9.8
   }
   ```

**预测时：**
1. 加载 `stats.json`
2. 使用训练时的统计信息归一化：
   ```python
   normalized = ((value - threshold) / (max - min)) + 0.5
   ```
3. 将阈值映射到 0.5，其他值按比例缩放
4. 最终分数范围：[0, 1]

**为什么阈值是 0.5？**
- 训练时的最优阈值被映射到 0.5
- 预测时 ≥ 0.5 表示超过训练阈值 → NG
- 预测时 < 0.5 表示低于训练阈值 → OK

---

## 📊 工作流程图

```
┌─────────────────────────────────────────────────────────┐
│                     训练阶段                            │
└─────────────────────────────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │    加载数据集 (good + defect)    │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  [可选] PCB 背景裁剪            │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │    调整为 512×512               │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  切分成 2×2 = 4 个 tiles        │
         │  (每个 256×256)                 │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  每个 tile 训练独立模型         │
         │  - model0_0.ckpt (左上)         │
         │  - model0_1.ckpt (右上)         │
         │  - model1_0.ckpt (左下)         │
         │  - model1_1.ckpt (右下)         │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  保存统计信息 (stats.json)      │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │    测试评估                     │
         └─────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     预测阶段                            │
└─────────────────────────────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  加载模型和统计信息             │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  [可选] 选择 ROI 区域           │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  读取图像                       │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  [可选] PCB 背景裁剪            │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  调整为 512×512                 │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  切分成 4 个 tiles              │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  每个 tile 用对应模型预测       │
         │  - Tile 0 → model0_0.ckpt       │
         │  - Tile 1 → model0_1.ckpt       │
         │  - Tile 2 → model1_0.ckpt       │
         │  - Tile 3 → model1_1.ckpt       │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  合并预测结果                   │
         │  - 拼接 anomaly_map             │
         │  - 平均 pred_score              │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  [可选] 接缝平滑                │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  归一化到 [0, 1]                │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  [可选] 应用 ROI mask           │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  判断 OK/NG                     │
         │  (分数 ≥ 0.5 → NG)             │
         └─────────────────────────────────┘
                           │
                           ↓
         ┌─────────────────────────────────┐
         │  生成可视化结果                 │
         └─────────────────────────────────┘
```

---

## 🛠️ 故障排查

### 问题 1: 找不到模型文件

**错误信息：**
```
❌ 找不到模型: results/PatchCore/pcb/latest/weights/lightning/model0_0.ckpt
```

**解决方法：**
1. 检查训练是否成功完成
2. 确认 `predict_config.yaml` 中的 `checkpoint_dir` 路径正确
3. 检查模型目录是否包含所有 4 个模型文件

### 问题 2: 找不到 stats.json

**错误信息：**
```
❌ 找不到 stats.json: results/PatchCore/pcb/latest/weights/lightning/stats.json
```

**解决方法：**
1. 重新运行训练（`stats.json` 在训练时生成）
2. 确认训练完全完成（未中断）

### 问题 3: PCB 裁剪失败

**警告信息：**
```
⚠️  PCB裁剪失败，使用原图
```

**可能原因：**
- 图像背景不是黑色
- PCB 边缘不明显
- 图像对比度过低

**解决方法：**
1. 调整 `min_area_ratio` 参数（降低到 0.05）
2. 禁用 PCB 裁剪：`pcb_crop.enable: false`

### 问题 4: ROI 选择窗口不显示

**可能原因：**
- OpenCV 窗口被其他窗口遮挡
- 显示器设置问题

**解决方法：**
1. 检查任务栏是否有 OpenCV 窗口
2. 按 Alt+Tab 切换窗口
3. 如果无法使用 ROI，禁用该功能：`roi.enable: false`

### 问题 5: GPU 内存不足

**错误信息：**
```
CUDA out of memory
```

**解决方法：**
1. 减少 `batch_size`（在 `ens_config.yaml` 中）
2. 使用 CPU：在 `predict_config.yaml` 中设置 `device: "cpu"`
3. 关闭其他占用 GPU 的程序

---

## 📝 注意事项

1. **训练前必读：**
   - 确保正常样本足够多（建议 > 100 张）
   - 异常样本仅用于测试，不参与训练
   - 图像尺寸最好接近 512×512（或更大）

2. **预测前必读：**
   - 必须先完成训练
   - 确保 `stats.json` 和 4 个模型文件都存在
   - 预测图像应与训练图像来自同一场景/设备

3. **参数调整建议：**
   - 如果误报多：提高 `normalized_threshold`（如 0.6）
   - 如果漏检多：降低 `normalized_threshold`（如 0.4）
   - 如果接缝明显：增大 `seam_smoothing.width`（如 0.15）

4. **性能优化：**
   - 使用 GPU 可显著加速（推荐 CUDA）
   - 预测时可以批量处理多张图像
   - 如果图像很大，考虑增加 `tile_size` 到 512×512

---

## 📚 参考资料

- [Anomalib 官方文档](https://anomalib.readthedocs.io/)
- [PatchCore 论文](https://arxiv.org/abs/2106.08265)
- [Tiled Ensemble 设计思想](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/pipelines/tiled_ensemble.html)

---

## 🤝 贡献与反馈

如有问题或建议，请联系项目维护者。

---

**最后更新：** 2026-01-12

