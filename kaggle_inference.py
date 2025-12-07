"""
Kaggle Notebook 推理脚本
直接复制到 Kaggle Notebook cell 中运行

前提条件:
1. 上传 src/ 文件夹到 Kaggle (包含 model.py, dataset.py 等)
2. 上传 output/ 文件夹到 Kaggle (包含 best_model.pth)
3. 添加比赛数据集

使用方法:
1. 在 Kaggle Notebook 中创建 Code Cell
2. 复制粘贴整个脚本
3. 运行即可生成 submission.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# ============================================
# 配置区域 - 根据实际情况修改
# ============================================

# Kaggle 中的路径配置
MODEL_WEIGHT_PATH = '/kaggle/input/your-dataset-name/output/best_model.pth'  # 修改为实际路径
SRC_PATH = '/kaggle/input/your-dataset-name/src'  # 修改为实际路径
TEST_CSV_PATH = '/kaggle/input/csiro-biomass/test.csv'  # 比赛数据集路径
TEST_IMG_ROOT = '/kaggle/input/csiro-biomass'  # 测试图片根目录
OUTPUT_PATH = 'submission.csv'

# 模型配置
MODEL_NAME = 'tf_efficientnet_b0'
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

# 目标列名
TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

# ============================================
# 导入模型模块
# ============================================

# 添加 src 到 Python 路径
sys.path.insert(0, SRC_PATH)

try:
    from model import BiomassModel
    print("✓ 成功导入模型模块")
except ImportError as e:
    print(f"✗ 导入模型失败: {e}")
    print("请确保 src 文件夹已正确上传到 Kaggle")
    sys.exit(1)

# ============================================
# 数据预处理
# ============================================

def get_test_transforms(image_size=224):
    """获取测试集图像变换"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class TestDataset(torch.utils.data.Dataset):
    """测试集数据加载器"""
    
    def __init__(self, image_paths, root_dir, transform=None):
        self.image_paths = image_paths
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        full_img_path = os.path.join(self.root_dir, img_path)
        
        try:
            image = Image.open(full_img_path).convert('RGB')
        except Exception as e:
            print(f"⚠ Error loading {full_img_path}: {e}")
            # 返回黑色图像作为备用
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path


# ============================================
# 主推理流程
# ============================================

def main():
    print("=" * 60)
    print("Kaggle Biomass Prediction Inference")
    print("=" * 60)
    
    # 1. 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n1️⃣ 使用设备: {device}")
    
    # 2. 加载模型
    print(f"\n2️⃣ 加载模型...")
    model = BiomassModel(
        model_name=MODEL_NAME,
        pretrained=False,
        num_classes=5
    )
    
    # 加载权重
    if not os.path.exists(MODEL_WEIGHT_PATH):
        print(f"✗ 模型文件不存在: {MODEL_WEIGHT_PATH}")
        print("请检查路径是否正确")
        return
    
    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功")
    if 'val_rmse' in checkpoint:
        print(f"  训练时验证集 RMSE: {checkpoint['val_rmse']:.4f}")
    
    # 3. 加载测试集
    print(f"\n3️⃣ 加载测试集...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    print(f"✓ 测试集行数: {len(test_df)}")
    
    # 获取唯一图片路径
    unique_image_paths = test_df['image_path'].unique()
    print(f"✓ 唯一图片数: {len(unique_image_paths)}")
    
    # 4. 创建数据加载器
    print(f"\n4️⃣ 创建数据加载器...")
    test_dataset = TestDataset(
        image_paths=unique_image_paths,
        root_dir=TEST_IMG_ROOT,
        transform=get_test_transforms(IMAGE_SIZE)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 5. 进行预测
    print(f"\n5️⃣ 开始预测...")
    predictions = {}
    
    with torch.no_grad():
        for images, img_paths in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.cpu().numpy()
            
            # 保存每张图片的预测结果
            for i, img_path in enumerate(img_paths):
                predictions[img_path] = outputs[i]
    
    print(f"✓ 完成预测，共 {len(predictions)} 张图片")
    
    # 6. 生成提交文件
    print(f"\n6️⃣ 生成提交文件...")
    submission_data = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Creating submission'):
        sample_id = row['sample_id']
        image_path = row['image_path']
        target_name = row['target_name']
        
        # 获取该图像的预测向量
        if image_path in predictions:
            pred_vector = predictions[image_path]
            # 根据 target_name 选择对应的值
            target_idx = TARGET_COLS.index(target_name)
            pred_value = pred_vector[target_idx]
        else:
            print(f"⚠ 警告: 未找到图片 {image_path} 的预测结果")
            pred_value = 0.0
        
        submission_data.append({
            'sample_id': sample_id,
            'target': pred_value
        })
    
    # 创建 DataFrame
    submission_df = pd.DataFrame(submission_data)
    
    # 保存
    submission_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✅ 提交文件已保存: {OUTPUT_PATH}")
    print(f"   文件形状: {submission_df.shape}")
    print(f"\n前 10 行预览:")
    print(submission_df.head(10))
    print(f"\n预测值统计:")
    print(submission_df['target'].describe())
    
    print("\n" + "=" * 60)
    print("推理完成！")
    print("=" * 60)
    
    return submission_df


# ============================================
# 运行
# ============================================

if __name__ == "__main__":
    submission_df = main()
