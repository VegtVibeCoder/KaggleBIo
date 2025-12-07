# 🌿 Kaggle Pasture Biomass Prediction - Baseline

## 1. 项目简介 (Overview)

本项目是 Kaggle 牧场生物量预测比赛的 **Baseline 代码库**。由于测试集 (Test Set) 缺失 NDVI、高度等元数据，本 Baseline 采用 **纯视觉 (Pure Vision)** 策略。

* **输入:** 仅使用图像 (RGB Images)
* **输出:** 同时预测 5 个生物量指标 (Multi-output Regression)
* **模型:** `timm` (EfficientNet_B0 / ResNet18)

---

## 2. 目录结构 (Directory Structure)

```text
├── data/                    # 【Git 忽略】
│   ├── raw/                 # 原始只读数据 (train.csv, images/)
│   └── processed/           # 预处理后的中间数据 (train_pivot.csv)
│
├── output/                  # 【Git 忽略】 <--- 你指出的缺失部分！
│   ├── checkpoints/         # 存放训练好的模型权重 (.pth)
│   ├── logs/                # TensorBoard 或 WandB 的日志
│   └── submissions/         # 生成的 csv 提交文件
│
├── configs/                 # 存放配置文件
│   └── baseline_v1.yaml     # 例如：定义 batch_size, lr, backbone
│
├── src/                     # 核心代码库
│   ├── dataset.py           # 数据定义
│   ├── model.py             # 模型定义
│   ├── train.py             # 训练逻辑
│   ├── inference.py         # 推理逻辑
│   ├── utils.py             # 工具函数 (seed_everything, metric计算)
│   └── loss.py              # 自定义 Loss (如果有)
│
├── notebooks/               # 实验性草稿
│   └── 01_eda_check_data.ipynb
│
├── requirements.txt         # 环境依赖
└── README.md                # 项目文档
```

---

## 3. Baseline 执行路线图

我们将整个流程分为 4 个明确的阶段：

### Phase 1: 数据清洗 (Data Prep)

* **目标:** 将原始的"长表" (`train.csv`) 转换为适合神经网络训练的"宽表"
* **动作:**
  1. 使用 Pandas `pivot` 功能，让每一行代表一张唯一的图片
  2. 生成 5 个新列：`['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']`
  3. 处理缺失值（如果有图片缺少某种生物量测量，暂时填 0 或平均值，保证程序不崩）

### Phase 2: 数据管道 (Dataset & Dataloader)

* **目标:** 构建 PyTorch `Dataset` 类
* **动作:**
  1. 输入：图片路径
  2. 输出：`(Image_Tensor, Target_Vector_of_Size_5)`
  3. 增加基础的数据增强（Resize 到 224x224, Normalize）

### Phase 3: 模型训练 (Training)

* **目标:** 跑通一个简单的 CNN
* **配置:**
  * **Backbone:** `resnet18` 或 `efficientnet_b0` (使用 `timm` 库，`pretrained=True`)
  * **Head:** 修改最后一层全连接层，输出维度 `num_classes=5`
  * **Loss:** `MSELoss` (均方误差) 或 `L1Loss`
  * **Metric:** 监控 RMSE

### Phase 4: 推理与提交 (Inference)

* **目标:** 生成 `submission.csv`
* **逻辑:**
  1. 加载 `test.csv`
  2. 对于每一行图片，模型预测出 5 个值
  3. 根据这一行的 `target_name`，从 5 个值里"查表"取出对应的那个值
  4. 保存文件

---

## 4. 核心流程详解 (Workflow)

### 4.1 数据预处理 (Data Transformation)

原始 `train.csv` 是长格式 (Long Format)，即一张图片对应多行数据。我们需要将其转换为宽格式 (Wide Format)。

**逻辑示例:**

> 转换前:
> ```
> img1.jpg | Dry_Green | 10
> img1.jpg | Dry_Dead  | 5
> ```
>
> 转换后:
> ```
> img1.jpg | [10, 5, ...] (Target Vector)
> ```

在 `train.py` 开始前，我们会执行以下操作：

1. 读取 `train.csv`
2. 执行 `df.pivot(index='image_path', columns='target_name', values='target')`
3. 确保所有 NaN 值被处理（填充 0 或均值）

### 4.2 模型架构 (Model Architecture)

使用单一 CNN Backbone 处理图像，输出层修改为 5 个神经元。

```python
class BiomassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnet_b0', 
            pretrained=True, 
            num_classes=5
        )
    
    def forward(self, x):
        return self.backbone(x)  # Output shape: [Batch_Size, 5]
```

### 4.3 推理与提交逻辑 (Inference Logic)

这是生成 `submission.csv` 的关键步骤。由于测试集每一行只需要一个特定的生物量，我们需要**按需取值**。

**操作流程:**

1. 读取 `test.csv`
2. 对唯一的 `image_path` 进行去重，批量输入模型进行预测
3. 得到该图片的 5 个预测值：`{Green: 10, Dead: 5, Clover: 2, ...}`
4. 回到 `test.csv` 的每一行：
   - 如果 `target_name == 'Dry_Dead_g'`，则填入 `5`
   - 如果 `target_name == 'Dry_Green_g'`，则填入 `10`

---

## 5. 快速开始 (Quick Start)

### 5.1 环境安装

```bash
pip install torch torchvision timm pandas opencv-python
```

### 5.2 运行训练

```bash
python src/train.py --epochs 10 --batch_size 32 --model efficientnet_b0
```

> 这将生成 `best_model.pth` 保存在 `output/` 目录

### 5.3 运行推理

```bash
python src/inference.py --weights output/best_model.pth
```

> 这将生成 `submission.csv`

---

## 6. 待办事项 (To-Do List)

- [ ] **EDA:** 检查是否有图片缺失了全部 5 个标签？检查 Label 分布是否长尾？
- [ ] **Validation:** 实现 GroupKFold (按 Location 或 Date 切分)，防止 Leakage
- [ ] **Augmentation:** 增加 Flip, Rotate, ColorJitter 等增强
- [ ] **Future Work:** 尝试 Pseudo-Labeling 以利用训练集的 Metadata
