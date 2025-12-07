# 使用指南 (Usage Guide)

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保数据目录结构如下：

```
csiro-biomass/
├── train.csv
├── test.csv
├── train/              # 训练图像
│   ├── ID1011485656.jpg
│   └── ...
└── test/               # 测试图像
    └── ...
```

### 3. 运行完整流程

#### 方式一：使用脚本（推荐）

```bash
./run_baseline.sh
```

#### 方式二：分步执行

**Step 1: 训练模型**

```bash
python src/train.py \
    --model tf_efficientnet_b0 \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001 \
    --pretrained
```

**Step 2: 生成提交文件**

```bash
python src/inference.py \
    --test_csv csiro-biomass/test.csv \
    --weights output/best_model.pth \
    --output submission.csv
```

---

## 详细参数说明

### 训练参数 (train.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `data` | 数据目录 |
| `--model` | `tf_efficientnet_b0` | 模型架构 (可选: resnet18, resnet34, resnet50) |
| `--epochs` | `20` | 训练轮数 |
| `--batch_size` | `32` | 批次大小 |
| `--lr` | `0.001` | 学习率 |
| `--image_size` | `224` | 图像大小 |
| `--loss` | `mse` | 损失函数 (mse 或 l1) |
| `--pretrained` | `True` | 使用预训练权重 |
| `--output_dir` | `output` | 输出目录 |

### 推理参数 (inference.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--test_csv` | `csiro-biomass/test.csv` | 测试集 CSV |
| `--weights` | `output/best_model.pth` | 模型权重路径 |
| `--model` | `tf_efficientnet_b0` | 模型架构 |
| `--batch_size` | `32` | 批次大小 |
| `--output` | `submission.csv` | 输出提交文件 |

---

## 实验建议

### 1. 不同模型对比

```bash
# EfficientNet-B0 (快速)
python src/train.py --model tf_efficientnet_b0 --epochs 20

# ResNet18 (轻量)
python src/train.py --model resnet18 --epochs 20

# ResNet50 (更强)
python src/train.py --model resnet50 --epochs 30 --batch_size 16
```

### 2. 超参数调优

```bash
# 更大的学习率
python src/train.py --lr 0.003 --epochs 15

# L1 损失 (对异常值更鲁棒)
python src/train.py --loss l1 --epochs 20

# 更大的图像尺寸
python src/train.py --image_size 256 --batch_size 16
```

### 3. 训练时间估算

- **EfficientNet-B0**: 约 10-15 分钟/epoch (GPU)
- **ResNet18**: 约 5-8 分钟/epoch (GPU)
- **ResNet50**: 约 15-20 分钟/epoch (GPU)

---

## 常见问题

### Q: 内存不足 (Out of Memory)

**解决方案:**
- 减小 batch_size: `--batch_size 16` 或 `--batch_size 8`
- 减小图像尺寸: `--image_size 192`
- 使用更小的模型: `--model resnet18`

### Q: 训练速度慢

**解决方案:**
- 确保使用 GPU: 检查 `torch.cuda.is_available()`
- 增加 num_workers: `--num_workers 8`
- 使用更小的模型

### Q: 模型不收敛

**解决方案:**
- 降低学习率: `--lr 0.0001`
- 增加训练轮数: `--epochs 50`
- 尝试不同的损失函数: `--loss l1`

---

## 输出文件

训练完成后，会生成以下文件：

```
output/
├── best_model.pth           # 最佳模型 (验证集 RMSE 最低)
├── last_model.pth           # 最后一轮的模型
└── training_history.png     # 训练曲线图

submission.csv               # 提交文件
```

---

## 下一步改进方向

1. **数据增强**: 在 `dataset.py` 中添加更多增强方法
2. **交叉验证**: 实现 K-Fold 验证
3. **模型集成**: 训练多个模型并平均预测结果
4. **使用元数据**: 如果可用，结合 NDVI、高度等特征
5. **后处理**: 添加预测值的后处理逻辑

---

## 技术支持

如有问题，请检查：
1. 数据路径是否正确
2. Python 环境和依赖版本
3. GPU 驱动和 CUDA 版本

Happy Coding! 🌿
