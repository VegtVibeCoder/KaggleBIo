"""
数据加载和预处理模块
包含数据转换（长表转宽表）和 PyTorch Dataset 类
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def prepare_data(csv_path, output_path=None):
    """
    将长格式的训练数据转换为宽格式
    
    Args:
        csv_path: 原始 train.csv 路径
        output_path: 输出的宽表路径（可选）
    
    Returns:
        df_wide: 宽格式 DataFrame
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 查看数据基本信息
    print(f"Original data shape: {df.shape}")
    print(f"Unique images: {df['image_path'].nunique()}")
    
    # 使用 pivot 将长表转为宽表
    df_wide = df.pivot(
        index='image_path',
        columns='target_name',
        values='target'
    ).reset_index()
    
    # 确保列的顺序一致
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    df_wide = df_wide[['image_path'] + target_cols]
    
    # 检查并处理缺失值
    if df_wide[target_cols].isnull().any().any():
        print("Warning: Found NaN values, filling with 0")
        df_wide[target_cols] = df_wide[target_cols].fillna(0)
    
    print(f"Wide format data shape: {df_wide.shape}")
    print(f"Columns: {df_wide.columns.tolist()}")
    
    # 保存处理后的数据
    if output_path:
        df_wide.to_csv(output_path, index=False)
        print(f"Saved wide format data to {output_path}")
    
    return df_wide


class BiomassDataset(Dataset):
    """
    牧场生物量预测数据集
    
    输入: RGB 图像
    输出: 5 个生物量指标的向量
    """
    
    def __init__(self, csv_path, root_dir, transform=None, is_train=True):
        """
        Args:
            csv_path: CSV 文件路径（宽格式）
            root_dir: 图像根目录
            transform: 图像变换
            is_train: 是否为训练模式
        """
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # 目标列名
        self.target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 获取图像路径
        img_path = self.df.iloc[idx]['image_path']
        full_img_path = os.path.join(self.root_dir, img_path)
        
        # 加载图像
        try:
            image = Image.open(full_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_img_path}: {e}")
            # 返回黑色图像作为备用
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        if self.is_train:
            # 获取目标值
            targets = self.df.iloc[idx][self.target_cols].values.astype(np.float32)
            targets = torch.from_numpy(targets)
            return image, targets
        else:
            return image, img_path


def get_transforms(image_size=224, is_train=True):
    """
    获取图像变换
    
    Args:
        image_size: 图像大小
        is_train: 是否为训练模式
    
    Returns:
        transforms: torchvision transforms
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


if __name__ == "__main__":
    # 测试数据预处理
    train_csv = "../csiro-biomass/train.csv"
    output_csv = "../data/processed/train_pivot.csv"
    
    if os.path.exists(train_csv):
        df_wide = prepare_data(train_csv, output_csv)
        print("\nFirst few rows:")
        print(df_wide.head())
        print("\nStatistics:")
        print(df_wide.describe())
