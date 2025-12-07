# 安装指南

## 系统要求

- Python 3.7+
- 8GB+ RAM (16GB 推荐)
- 10GB+ 磁盘空间
- GPU (可选，但强烈推荐)

## 安装步骤

### 1. 克隆或下载项目

```bash
cd /Users/carson/Desktop/code/CSIRO
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 venv
python3 -m venv venv
source venv/bin/activate

# 或使用 conda
conda create -n biomass python=3.9
conda activate biomass
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python3 test_setup.py
```

应该看到类似输出：
```
✓ 所有测试通过！可以开始训练了。
通过: 6/6 项测试
```

## GPU 支持（可选）

### CUDA (NVIDIA GPU)

如果有 NVIDIA GPU，安装 CUDA 版本的 PyTorch：

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证 CUDA：
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### MPS (Apple Silicon Mac)

Apple M1/M2 芯片自动支持 MPS 加速，无需额外配置。

## 依赖包说明

| 包 | 版本 | 用途 |
|---|------|------|
| torch | 2.0+ | 深度学习框架 |
| torchvision | 0.15+ | 图像处理 |
| timm | 0.9+ | 预训练模型库 |
| pandas | 2.0+ | 数据处理 |
| numpy | 1.24+ | 数值计算 |
| opencv-python | 4.8+ | 图像处理 |
| pillow | 10.0+ | 图像加载 |
| scikit-learn | 1.3+ | 数据分割 |
| tqdm | 4.65+ | 进度条 |
| matplotlib | 3.7+ | 可视化 |

## 常见问题

### Q: pip install 速度慢

使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: ImportError: No module named 'xxx'

重新安装依赖：

```bash
pip install -r requirements.txt --upgrade
```

### Q: CUDA out of memory

这不是安装问题，是训练时的内存问题。请减小 batch_size。

## 下一步

安装完成后：

1. 运行测试：`python3 test_setup.py`
2. 查看快速开始：`cat QUICKSTART.md`
3. 开始训练：`./run_baseline.sh`

## 卸载

```bash
# 删除虚拟环境
deactivate
rm -rf venv

# 或 conda
conda deactivate
conda remove -n biomass --all
```

---

**需要帮助？** 查看 `USAGE.md` 或 `PROJECT_SUMMARY.md`
