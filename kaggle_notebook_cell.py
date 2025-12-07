"""
📋 Kaggle Notebook 推理脚本 - 单 Cell 版本
直接复制整个文件内容到 Kaggle Notebook 的一个 Code Cell 中运行

使用前请修改以下路径:
1. MODEL_WEIGHT_PATH - 你上传的模型权重路径
2. SRC_PATH - 你上传的 src 文件夹路径
"""

# ==================== 开始复制 ====================

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# 🔧 配置区 - 请根据实际情况修改
MODEL_WEIGHT_PATH = '/kaggle/input/csiro-model/output/best_model.pth'  # 👈 修改这里
SRC_PATH = '/kaggle/input/csiro-model/src'  # 👈 修改这里
TEST_CSV_PATH = '/kaggle/input/csiro-pasture-biomass-prediction/test.csv'
TEST_IMG_ROOT = '/kaggle/input/csiro-pasture-biomass-prediction'

MODEL_NAME = 'tf_efficientnet_b0'
IMAGE_SIZE = 224
BATCH_SIZE = 32

# 添加 src 到路径
sys.path.insert(0, SRC_PATH)
from model import BiomassModel

# 数据加载器
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, root_dir, transform):
        self.image_paths = image_paths
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        full_path = os.path.join(self.root_dir, img_path)
        try:
            image = Image.open(full_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        return self.transform(image), img_path

# 图像变换
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🖥️  Device: {device}')

model = BiomassModel(MODEL_NAME, pretrained=False, num_classes=5)
checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device).eval()
print(f'✅ Model loaded (Val RMSE: {checkpoint.get("val_rmse", "N/A"):.4f})')

# 加载测试集
test_df = pd.read_csv(TEST_CSV_PATH)
unique_imgs = test_df['image_path'].unique()
print(f'📊 Test images: {len(unique_imgs)}')

# 创建数据加载器
dataset = TestDataset(unique_imgs, TEST_IMG_ROOT, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 预测
predictions = {}
with torch.no_grad():
    for images, img_paths in tqdm(loader, desc='Predicting'):
        outputs = model(images.to(device)).cpu().numpy()
        for i, path in enumerate(img_paths):
            predictions[path] = outputs[i]

# 生成提交文件
TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
submission = []

for _, row in test_df.iterrows():
    pred_vector = predictions.get(row['image_path'], np.zeros(5))
    target_idx = TARGET_COLS.index(row['target_name'])
    submission.append({
        'sample_id': row['sample_id'],
        'target': pred_vector[target_idx]
    })

submission_df = pd.DataFrame(submission)
submission_df.to_csv('submission.csv', index=False)

print(f'\n✅ Submission saved!')
print(f'Shape: {submission_df.shape}')
print(submission_df.head(10))
print(submission_df['target'].describe())

# ==================== 结束复制 ====================
