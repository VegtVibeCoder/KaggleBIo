# 🌿 CSIRO Pasture Biomass Prediction

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/csiro-pasture-biomass-prediction)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)

## 📋 项目简介

本项目是 **Kaggle CSIRO Pasture Biomass Prediction** 比赛的完整解决方案。通过深度学习方法，从牧场 RGB 图像预测 5 个生物量指标。

### 🎯 任务目标

**任务类型**: 多输出回归 (Multi-output Regression)

从牧场图像预测以下 5 个生物量指标（单位：克）：
- `Dry_Green_g` - 干燥绿色植物重量
- `Dry_Dead_g` - 干燥枯死植物重量  
- `Dry_Clover_g` - 干燥三叶草重量
- `GDM_g` - 绿色干物质 (Green Dry Matter)
- `Dry_Total_g` - 总干物质重量

**评估指标**: RMSE (Root Mean Square Error)

### 🔑 核心特点

- ✅ **纯视觉方案** - 仅使用 RGB 图像（测试集无元数据）
- ✅ **端到端训练** - 单一模型同时预测 5 个指标
- ✅ **配置化管理** - 所有超参数集中在 `config.py`
- ✅ **Kaggle 就绪** - 提供完整的 Kaggle Notebook 推理脚本
- ✅ **模块化设计** - 清晰的代码结构，易于扩展

---

## 📁 项目结构

```text
CSIRO/
├── csiro-biomass/              # 数据集目录
│   ├── train/                  # 训练集图像
│   ├── test/                   # 测试集图像
│   ├── train.csv               # 训练标签（长格式）
│   ├── test.csv                # 测试集信息
│   └── sample_submission.csv   # 提交样例
│
├── src/                        # 核心代码库 ⭐
│   ├── __init__.py             # 包初始化
│   ├── config.py               # 配置管理（超参数、路径等）
│   ├── dataset.py              # 数据加载和预处理
│   ├── model.py                # 模型定义
│   ├── train.py                # 训练脚本（命令行参数）
│   ├── train_with_config.py    # 训练脚本（使用 config.py）
│   └── inference.py            # 推理脚本
│
├── output/                     # 训练输出 [Git 忽略]
│   ├── best_model.pth          # 最佳模型权重
│   ├── last_model.pth          # 最后一轮模型
│   ├── training_history.png    # 训练曲线图
│   └── logs/                   # 训练日志
│
├── data/                       # 数据处理中间文件 [Git 忽略]
│   └── processed/
│       └── train_pivot.csv     # 转换后的宽格式数据
│
├── notebooks/                  # Jupyter Notebooks
│   └── eda.py                  # 数据探索分析
│
├── kaggle_notebook_cell.py     # Kaggle Notebook 推理脚本 ⭐
├── prepare_kaggle_upload.sh    # Kaggle 上传准备脚本
├── requirements.txt            # Python 依赖
├── README.md                   # 项目文档
└── .gitignore                  # Git 忽略配置
```

---

## 📊 数据集说明

### 数据来源

**Kaggle Competition**: [CSIRO Pasture Biomass Prediction](https://www.kaggle.com/competitions/csiro-pasture-biomass-prediction)

### 数据格式

#### 训练集 (`train.csv`)

**格式**: 长格式 (Long Format) - 每张图片对应 5 行数据

| 列名 | 说明 | 示例 |
|------|------|------|
| `image_path` | 图像相对路径 | `train/image_001.jpg` |
| `target_name` | 目标指标名称 | `Dry_Green_g` |
| `target` | 目标值（克） | `125.3` |

**示例数据**:
```
image_path          | target_name    | target
--------------------|----------------|--------
train/img1.jpg      | Dry_Green_g    | 120.5
train/img1.jpg      | Dry_Dead_g     | 45.2
train/img1.jpg      | Dry_Clover_g   | 15.8
train/img1.jpg      | GDM_g          | 98.3
train/img1.jpg      | Dry_Total_g    | 181.5
```

#### 测试集 (`test.csv`)

**格式**: 每行一个预测任务

| 列名 | 说明 |
|------|------|
| `sample_id` | 提交 ID（格式：`图像ID__目标名称`）|
| `image_path` | 图像路径 |
| `target_name` | 需要预测的目标 |

**示例**:
```
sample_id                        | image_path        | target_name
---------------------------------|-------------------|-------------
ID1001187975__Dry_Green_g        | test/img_001.jpg  | Dry_Green_g
ID1001187975__Dry_Dead_g         | test/img_001.jpg  | Dry_Dead_g
```

### 数据统计

- **训练图像数**: ~2000+ 张
- **测试图像数**: ~500+ 张
- **图像尺寸**: 不固定（需要 resize）
- **图像格式**: JPG
- **目标指标**: 5 个连续值

### 数据预处理

**关键步骤**: 将长格式转换为宽格式

```python
# 转换前（长格式）- 一张图 5 行
image_path    | target_name  | target
img1.jpg      | Dry_Green_g  | 120.5
img1.jpg      | Dry_Dead_g   | 45.2
...

# 转换后（宽格式）- 一张图 1 行
image_path | Dry_Green_g | Dry_Dead_g | Dry_Clover_g | GDM_g | Dry_Total_g
img1.jpg   | 120.5       | 45.2       | 15.8         | 98.3  | 181.5
```

这样每张图像对应一个 5 维向量，适合神经网络训练。

---

## 🧠 项目思路与方法

### 整体架构

```
输入图像 (RGB) → CNN Backbone → 全连接层 → 5 个输出值
                  ↓
              特征提取
                  ↓
         [Dry_Green, Dry_Dead, Dry_Clover, GDM, Dry_Total]
```

### 技术方案

#### 1️⃣ 模型架构

**Backbone 选择** (使用 `timm` 库):
- `tf_efficientnet_b0` - 轻量级，快速训练（默认）
- `tf_efficientnet_b3` - 更高精度
- `convnext_tiny` - 现代架构
- `swin_transformer` - 最高精度（需要更多资源）

**输出层设计**:
```python
class BiomassModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0'):
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,  # 使用 ImageNet 预训练权重
            num_classes=5     # 输出 5 个值
        )
```

#### 2️⃣ 训练策略

**损失函数**: MSE Loss (均方误差)
```python
loss = MSELoss(predictions, targets)  # targets shape: [batch, 5]
```

**优化器**: AdamW
- Learning Rate: 1e-4
- Weight Decay: 1e-4

**学习率调度**: Cosine Annealing
- 平滑降低学习率
- 避免陷入局部最优

**数据增强**:
- 训练集: Resize, RandomFlip, ColorJitter, Normalize
- 验证集: Resize, Normalize

#### 3️⃣ 推理逻辑

**关键问题**: 测试集每行只需要一个值，但模型输出 5 个值

**解决方案**:
1. 对每张唯一图像预测一次，得到 5 个值
2. 根据 `target_name` 索引对应的值

```python
# 预测
predictions = model(image)  # [Dry_Green, Dry_Dead, Dry_Clover, GDM, Dry_Total]

# 根据 target_name 取值
if target_name == 'Dry_Green_g':
    result = predictions[0]
elif target_name == 'Dry_Dead_g':
    result = predictions[1]
# ...
```

### 工作流程

```
1. 数据预处理 → 2. 模型训练 → 3. 模型评估 → 4. Kaggle 推理 → 5. 生成提交
     ↓              ↓              ↓              ↓              ↓
  长转宽格式      训练+验证      保存最佳模型    加载模型       submission.csv
```

---

## 🗂️ src/ 文件夹详解

### `config.py` - 配置管理 ⭐

集中管理所有训练和推理参数，方便实验和调优。

**主要配置项**:
```python
# 模型配置
MODEL_NAME = 'tf_efficientnet_b0'
PRETRAINED = True
NUM_CLASSES = 5

# 训练配置
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
OPTIMIZER = 'adamw'
SCHEDULER = 'cosine'

# 数据增强
IMAGE_SIZE = 224
TRAIN_AUGMENTATION = {...}

# 高级功能
USE_AMP = True              # 混合精度训练
GRADIENT_CLIP = 1.0         # 梯度裁剪
EARLY_STOPPING = True       # 早停策略
```

**使用方法**:
```python
from config import cfg

model_name = cfg.MODEL_NAME
batch_size = cfg.BATCH_SIZE
```

### `dataset.py` - 数据加载和预处理

**核心组件**:

1. **`prepare_data()`** - 数据格式转换
   - 将长格式转换为宽格式
   - 处理缺失值
   
2. **`BiomassDataset`** - PyTorch Dataset 类
   - 加载图像和标签
   - 应用数据增强
   
3. **`get_transforms()`** - 数据增强
   - 训练集：Resize, Flip, ColorJitter, Normalize
   - 验证集：Resize, Normalize

**使用示例**:
```python
from dataset import BiomassDataset, get_transforms

dataset = BiomassDataset(
    csv_path='data/processed/train_pivot.csv',
    root_dir='csiro-biomass',
    transform=get_transforms(image_size=224, is_train=True)
)
```

### `model.py` - 模型定义

**核心类**:

1. **`BiomassModel`** - 主模型类
   - 使用 timm 库的预训练模型
   - 输出 5 个生物量指标
   
2. **`create_model()`** - 模型创建辅助函数
   - 自动移动到指定设备
   - 打印模型参数量

**使用示例**:
```python
from model import create_model

model = create_model(
    model_name='tf_efficientnet_b0',
    pretrained=True,
    device='cuda'
)
```

### `train_with_config.py` - 训练脚本（推荐）⭐

使用 `config.py` 配置的训练脚本，包含完整的训练流程。

**功能特性**:
- ✅ 自动从配置读取所有参数
- ✅ 支持混合精度训练（AMP）
- ✅ 梯度裁剪防止梯度爆炸
- ✅ 早停策略避免过拟合
- ✅ 学习率调度器
- ✅ 训练曲线可视化

**运行方法**:
```bash
# 1. 修改 src/config.py 中的参数
# 2. 运行训练
python src/train_with_config.py
```

### `train.py` - 训练脚本（命令行参数）

使用命令行参数的训练脚本，适合快速实验。

**运行方法**:
```bash
python src/train.py \
    --model tf_efficientnet_b0 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4
```

### `inference.py` - 推理脚本

生成 Kaggle 提交文件。

**运行方法**:
```bash
python src/inference.py \
    --weights output/best_model.pth \
    --test_csv csiro-biomass/test.csv \
    --output submission.csv
```

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone <your-repo-url>
cd CSIRO

# 安装依赖
pip install -r requirements.txt
```

**依赖包**:
- `torch >= 2.0.0`
- `torchvision`
- `timm` - 预训练模型库
- `pandas`
- `numpy`
- `Pillow`
- `tqdm`
- `matplotlib`

### 2. 准备数据

```bash
# 下载 Kaggle 数据集
kaggle competitions download -c csiro-pasture-biomass-prediction

# 解压到 csiro-biomass/ 目录
unzip csiro-pasture-biomass-prediction.zip -d csiro-biomass/
```

### 3. 训练模型

**方式 1: 使用配置文件（推荐）**

```bash
# 1. 编辑配置
vim src/config.py

# 2. 运行训练
python src/train_with_config.py
```

**方式 2: 使用命令行参数**

```bash
python src/train.py \
    --model tf_efficientnet_b0 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_dir output
```

### 4. 本地推理

```bash
python src/inference.py \
    --weights output/best_model.pth \
    --test_csv csiro-biomass/test.csv \
    --output submission.csv
```

---

## 📤 Kaggle 使用方法

### 方案概述

由于 Kaggle Notebook 环境限制，我们采用以下方案：

1. **本地训练** - 在本地/服务器训练模型
2. **上传资源** - 将模型和代码上传到 Kaggle Dataset
3. **Notebook 推理** - 在 Kaggle Notebook 中加载模型进行推理

### 详细步骤

#### Step 1: 本地训练模型

```bash
# 训练模型
python src/train_with_config.py

# 训练完成后，模型保存在 output/best_model.pth
```

#### Step 2: 准备上传文件

```bash
# 使用准备脚本
bash prepare_kaggle_upload.sh

# 或手动准备
mkdir kaggle_upload
cp -r src/ kaggle_upload/
cp output/best_model.pth kaggle_upload/
```

需要上传的文件：
- `src/` 文件夹（包含 `model.py` 等）
- `best_model.pth`（训练好的模型权重）

#### Step 3: 上传到 Kaggle Dataset

1. 访问 https://www.kaggle.com/datasets
2. 点击 "New Dataset"
3. 上传文件：
   - `src/` 文件夹 → 命名为 `srcorigin`
   - `best_model.pth` → 命名为 `best-model`
4. 发布 Dataset

#### Step 4: 在 Kaggle Notebook 中使用

**创建新的 Kaggle Notebook**:

1. 添加数据源：
   - 比赛数据集：`csiro-pasture-biomass-prediction`
   - 你的 Dataset：`srcorigin` 和 `best-model`

2. 复制 `kaggle_notebook_cell.py` 的内容到 Code Cell

3. **修改路径配置**（重要！）:

```python
# 在 kaggle_notebook_cell.py 中修改这些路径
MODEL_WEIGHT_PATH = '/kaggle/input/best-model/best_model.pth'
SRC_PATH = '/kaggle/input/srcorigin/src'
TEST_CSV_PATH = '/kaggle/input/csiro-biomass/test.csv'
TEST_IMG_ROOT = '/kaggle/input/csiro-biomass/test'
```

4. 运行 Cell，生成 `submission.csv`

5. 提交到比赛

### Kaggle Notebook 代码说明

`kaggle_notebook_cell.py` 包含完整的推理流程：

```python
# 1. 导入必要的库
import torch, pandas, numpy, ...

# 2. 配置路径
MODEL_WEIGHT_PATH = '/kaggle/input/...'
SRC_PATH = '/kaggle/input/...'

# 3. 添加 src 到路径
sys.path.insert(0, SRC_PATH)
from model import BiomassModel

# 4. 加载模型
model = BiomassModel(...)
checkpoint = torch.load(MODEL_WEIGHT_PATH, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# 5. 推理
for images in test_loader:
    predictions = model(images)

# 6. 生成提交文件
submission_df.to_csv('submission.csv', index=False)
```

### 常见问题

**Q: 为什么要用 `weights_only=False`？**

A: PyTorch 2.6+ 默认 `weights_only=True`，但我们的模型包含 numpy 对象，需要设置为 `False`。

**Q: 如何更新模型？**

A: 重新训练后，更新 Kaggle Dataset 中的 `best_model.pth` 文件即可。

**Q: 可以在 Kaggle 上训练吗？**

A: 可以，但由于时间限制（9小时），建议本地训练后上传。

---

## 📈 实验与优化

### 模型选择

| 模型 | 参数量 | 训练速度 | 预期精度 | 推荐场景 |
|------|--------|----------|----------|----------|
| `tf_efficientnet_b0` | 5M | ⚡⚡⚡ | ⭐⭐⭐ | 快速实验 |
| `tf_efficientnet_b1` | 7M | ⚡⚡ | ⭐⭐⭐⭐ | 平衡选择 |
| `tf_efficientnet_b3` | 12M | ⚡ | ⭐⭐⭐⭐⭐ | 高精度 |
| `convnext_tiny` | 28M | ⚡ | ⭐⭐⭐⭐⭐ | 现代架构 |
| `swin_tiny` | 28M | 🐌 | ⭐⭐⭐⭐⭐⭐ | 最高精度 |

### 超参数调优建议

**学习率**:
- 小模型（b0, b1）: `1e-3` ~ `5e-4`
- 大模型（b3, convnext）: `1e-4` ~ `5e-5`

**批量大小**:
- 根据显存调整：8, 16, 32, 64
- 大批量需要更高学习率

**数据增强**:
```python
# 轻度增强（推荐）
transforms.RandomHorizontalFlip(p=0.5)
transforms.ColorJitter(brightness=0.2, contrast=0.2)

# 重度增强（可能过拟合）
transforms.RandomRotation(30)
transforms.RandomAffine(...)
```

### 性能优化技巧

1. **混合精度训练** - 设置 `USE_AMP=True`，加速 2x
2. **梯度累积** - 显存不足时模拟大批量
3. **学习率预热** - 前几轮逐渐增加学习率
4. **TTA (Test Time Augmentation)** - 推理时多次增强取平均

---

## 📝 开发日志

### 已完成功能

- [x] 数据预处理（长转宽格式）
- [x] PyTorch Dataset 和 DataLoader
- [x] 模型定义（支持多种 backbone）
- [x] 训练脚本（支持配置文件和命令行）
- [x] 推理脚本
- [x] Kaggle Notebook 集成
- [x] 配置化管理系统
- [x] 混合精度训练
- [x] 早停策略
- [x] 训练可视化

### 待优化功能

- [ ] K-Fold 交叉验证
- [ ] 模型集成（Ensemble）
- [ ] TTA（测试时增强）
- [ ] 自动超参数搜索
- [ ] WandB 日志集成
- [ ] 更多数据增强策略

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发流程

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目仅供学习和研究使用。

---

## 🙏 致谢

- Kaggle CSIRO Pasture Biomass Prediction Competition
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- PyTorch Team

---

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

**Happy Coding! 🚀**
