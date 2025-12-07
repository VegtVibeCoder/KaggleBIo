# ⚡ 快速开始指南

## 🎯 3 分钟上手

### 步骤 1: 安装依赖 (1 分钟)

```bash
pip install -r requirements.txt
```

### 步骤 2: 测试环境 (1 分钟)

```bash
python3 test_setup.py
```

### 步骤 3: 开始训练 (1 分钟设置)

```bash
# 方式一：一键运行（推荐）
./run_baseline.sh

# 方式二：自定义参数
python3 src/train.py --epochs 10 --batch_size 32
```

---

## 📋 常用命令

### 训练

```bash
# 快速训练（10 轮）
python3 src/train.py --epochs 10 --batch_size 32

# 完整训练（20 轮）
python3 src/train.py --epochs 20 --batch_size 32 --pretrained

# 使用 ResNet18（更快）
python3 src/train.py --model resnet18 --epochs 15

# 小内存训练
python3 src/train.py --batch_size 16 --image_size 192
```

### 推理

```bash
# 生成提交文件
python3 src/inference.py \
    --weights output/best_model.pth \
    --output submission.csv
```

### 数据分析

```bash
# 运行 EDA
cd notebooks && python3 eda.py
```

---

## 🐛 常见问题

### Q: 内存不足 (OOM)

```bash
# 减小 batch size
python3 src/train.py --batch_size 8

# 或减小图像尺寸
python3 src/train.py --image_size 192
```

### Q: 训练太慢

```bash
# 使用更小的模型
python3 src/train.py --model resnet18

# 减少训练轮数
python3 src/train.py --epochs 10
```

### Q: 模型不收敛

```bash
# 降低学习率
python3 src/train.py --lr 0.0001

# 使用 L1 损失
python3 src/train.py --loss l1
```

---

## 📁 重要文件

| 文件 | 说明 |
|------|------|
| `README.md` | 完整项目文档 |
| `USAGE.md` | 详细使用指南 |
| `PROJECT_SUMMARY.md` | 项目总结 |
| `test_setup.py` | 环境测试 |

---

## 🎓 学习路径

1. **第一次使用**: 阅读 `README.md`
2. **深入了解**: 查看 `USAGE.md`
3. **开始训练**: 运行 `./run_baseline.sh`
4. **调试问题**: 运行 `python3 test_setup.py`
5. **数据分析**: 查看 `notebooks/eda.py`

---

## 💡 小贴士

- ✅ 第一次运行建议使用默认参数
- ✅ 训练前先运行 `test_setup.py` 检查环境
- ✅ 使用 GPU 可以加速 3-5 倍
- ✅ 保存好 `output/best_model.pth`
- ✅ 定期检查 `output/training_history.png`

---

**需要帮助?** 查看 `USAGE.md` 或 `PROJECT_SUMMARY.md`
