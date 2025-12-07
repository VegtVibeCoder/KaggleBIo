# 🚀 Kaggle Notebook 使用指南

## 步骤 1: 准备上传文件

### 1.1 创建上传文件夹

在本地创建一个文件夹用于上传到 Kaggle：

```bash
mkdir kaggle_upload
cp -r src kaggle_upload/
cp -r output kaggle_upload/
```

### 1.2 检查文件

确保包含以下文件：

```
kaggle_upload/
├── src/
│   ├── __init__.py
│   ├── model.py         # ✅ 必需
│   ├── dataset.py       # (可选)
│   └── ...
└── output/
    └── best_model.pth   # ✅ 必需
```

---

## 步骤 2: 上传到 Kaggle Dataset

### 2.1 压缩文件

```bash
cd kaggle_upload
zip -r csiro-model.zip src/ output/
```

### 2.2 上传到 Kaggle

1. 访问 https://www.kaggle.com/datasets
2. 点击 **"New Dataset"**
3. 上传 `csiro-model.zip`
4. 设置 Dataset 名称，例如：`csiro-model`
5. 点击 **"Create"**

### 2.3 记录 Dataset 路径

上传后，Dataset 路径会是：
```
/kaggle/input/csiro-model/
```

---

## 步骤 3: 在 Kaggle Notebook 中使用

### 3.1 创建新 Notebook

1. 进入比赛页面
2. 点击 **"Code"** → **"New Notebook"**

### 3.2 添加 Dataset

在 Notebook 右侧：
1. 点击 **"Add data"**
2. 搜索并添加你的 `csiro-model` dataset
3. 确保比赛数据集也已添加

### 3.3 粘贴推理代码

#### 方式一：使用完整版本

复制 `kaggle_inference.py` 的内容，并修改以下路径：

```python
# 修改这些路径
MODEL_WEIGHT_PATH = '/kaggle/input/csiro-model/output/best_model.pth'
SRC_PATH = '/kaggle/input/csiro-model/src'
TEST_CSV_PATH = '/kaggle/input/csiro-pasture-biomass-prediction/test.csv'
TEST_IMG_ROOT = '/kaggle/input/csiro-pasture-biomass-prediction'
```

#### 方式二：使用精简版本

直接复制 `kaggle_notebook_cell.py` 的全部内容到一个 Cell 中，修改前几行的路径即可。

---

## 步骤 4: 运行推理

### 4.1 运行 Cell

点击运行，会看到类似输出：

```
🖥️  Device: GPU
✅ Model loaded (Val RMSE: 12.3456)
📊 Test images: 800+
Predicting: 100%|██████████| 25/25 [00:10<00:00,  2.50it/s]
✅ Submission saved!
```

### 4.2 提交结果

1. 点击右上角 **"Save Version"**
2. 选择 **"Save & Run All (Commit)"**
3. 等待运行完成
4. 点击 **"Submit to Competition"**

---

## 📁 完整的文件路径示例

假设你的 Kaggle username 是 `yourname`，dataset 名称是 `csiro-model`：

### 上传的文件结构

```
/kaggle/input/csiro-model/
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── dataset.py
└── output/
    └── best_model.pth
```

### 比赛数据集路径

```
/kaggle/input/csiro-pasture-biomass-prediction/
├── test.csv
├── test/
│   └── *.jpg
└── sample_submission.csv
```

---

## 🔧 路径配置示例

在 Kaggle Notebook 中使用时，配置如下：

```python
# 你上传的模型和代码
MODEL_WEIGHT_PATH = '/kaggle/input/csiro-model/output/best_model.pth'
SRC_PATH = '/kaggle/input/csiro-model/src'

# 比赛数据集（根据实际比赛名称修改）
TEST_CSV_PATH = '/kaggle/input/csiro-pasture-biomass-prediction/test.csv'
TEST_IMG_ROOT = '/kaggle/input/csiro-pasture-biomass-prediction'
```

---

## 🎯 快速检查清单

在运行前确认：

- [ ] ✅ `src/model.py` 文件存在
- [ ] ✅ `output/best_model.pth` 文件存在
- [ ] ✅ Dataset 已成功上传到 Kaggle
- [ ] ✅ Notebook 中已添加你的 dataset
- [ ] ✅ Notebook 中已添加比赛数据集
- [ ] ✅ 路径配置正确（特别是 dataset 名称）
- [ ] ✅ 开启了 GPU 加速（Settings → Accelerator → GPU）

---

## 💡 常见问题

### Q1: ModuleNotFoundError: No module named 'timm'

Kaggle Notebook 默认包含 timm，如果报错，添加：

```python
!pip install timm
```

### Q2: FileNotFoundError: Model weight not found

检查路径是否正确，可以先运行：

```python
!ls -la /kaggle/input/csiro-model/output/
```

### Q3: 如何查看可用的数据集路径？

运行：

```python
!ls -la /kaggle/input/
```

### Q4: GPU 内存不足

减小 batch size：

```python
BATCH_SIZE = 16  # 或 8
```

---

## 📊 预期运行时间

- **CPU**: ~15-20 分钟
- **GPU**: ~3-5 分钟

---

## 🎓 进阶技巧

### 1. 使用 TTA (Test Time Augmentation)

在推理时使用多个增强版本并平均：

```python
# 在预测部分添加
predictions_list = []
for aug in [flip_h, flip_v, rotate]:
    outputs = model(aug(images))
    predictions_list.append(outputs)
final_pred = torch.stack(predictions_list).mean(0)
```

### 2. 模型集成

如果训练了多个模型：

```python
models = [model1, model2, model3]
ensemble_pred = []
for model in models:
    pred = model(images)
    ensemble_pred.append(pred)
final_pred = torch.stack(ensemble_pred).mean(0)
```

---

## 🔗 相关文件

- `kaggle_inference.py` - 完整推理脚本
- `kaggle_notebook_cell.py` - 精简单 Cell 版本
- `src/model.py` - 模型定义
- `output/best_model.pth` - 训练好的权重

---

**祝提交顺利！🎉**
