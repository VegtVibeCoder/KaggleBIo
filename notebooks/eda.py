"""
探索性数据分析 (EDA)
用于快速了解数据分布和特性
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# 设置工作目录
sys.path.append('..')

# 设置绘图风格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_explore_data():
    """加载并探索训练数据"""
    
    print("=" * 60)
    print("CSIRO Pasture Biomass Prediction - EDA")
    print("=" * 60)
    
    # 加载数据
    train_csv = '../csiro-biomass/train.csv'
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found")
        return
    
    df = pd.read_csv(train_csv)
    
    # 基本信息
    print("\n1. 数据集基本信息")
    print("-" * 60)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"\n唯一图片数: {df['image_path'].nunique()}")
    print(f"唯一 target_name 数: {df['target_name'].nunique()}")
    print(f"Target names: {df['target_name'].unique().tolist()}")
    
    # 查看前几行
    print("\n2. 前几行数据")
    print("-" * 60)
    print(df.head(10))
    
    # 检查缺失值
    print("\n3. 缺失值检查")
    print("-" * 60)
    print(df.isnull().sum())
    
    # 转换为宽格式
    print("\n4. 转换为宽格式")
    print("-" * 60)
    df_wide = df.pivot(
        index='image_path',
        columns='target_name',
        values='target'
    ).reset_index()
    print(f"宽格式形状: {df_wide.shape}")
    
    # 目标变量统计
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    
    print("\n5. 目标变量统计")
    print("-" * 60)
    print(df_wide[target_cols].describe())
    
    # 绘制分布图
    print("\n6. 绘制目标变量分布...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(target_cols):
        axes[i].hist(df_wide[col], bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{col} Distribution')
        axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(target_cols), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('target_distributions.png', dpi=150, bbox_inches='tight')
    print("✓ 分布图保存为 target_distributions.png")
    
    # 相关性分析
    print("\n7. 目标变量相关性")
    print("-" * 60)
    correlation = df_wide[target_cols].corr()
    print(correlation)
    
    # 绘制相关性热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Target Variables Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("✓ 相关性热图保存为 correlation_heatmap.png")
    
    # 元数据分析（如果有）
    if 'State' in df.columns:
        print("\n8. 州 (State) 分布")
        print("-" * 60)
        print(df.groupby('State')['image_path'].nunique())
    
    if 'Species' in df.columns:
        print("\n9. 物种 (Species) 分布")
        print("-" * 60)
        species_counts = df.groupby('Species')['image_path'].nunique().sort_values(ascending=False)
        print(species_counts.head(10))
    
    # 检查图像
    print("\n10. 检查图像文件")
    print("-" * 60)
    sample_img_path = df.iloc[0]['image_path']
    full_path = os.path.join('..', sample_img_path)
    
    if os.path.exists(full_path):
        img = Image.open(full_path)
        print(f"✓ 示例图像: {sample_img_path}")
        print(f"  尺寸: {img.size}")
        print(f"  模式: {img.mode}")
        
        # 显示几张示例图片
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(min(8, len(df_wide))):
            img_path = df_wide.iloc[i]['image_path']
            full_path = os.path.join('..', img_path)
            
            if os.path.exists(full_path):
                img = Image.open(full_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f'Image {i+1}', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
        print("✓ 示例图片保存为 sample_images.png")
    else:
        print(f"✗ 图像文件未找到: {full_path}")
    
    print("\n" + "=" * 60)
    print("EDA 完成!")
    print("=" * 60)
    
    return df, df_wide


if __name__ == "__main__":
    df, df_wide = load_and_explore_data()
