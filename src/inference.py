"""
推理脚本
加载训练好的模型，对测试集进行预测并生成提交文件
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from dataset import get_transforms
from model import create_model


class TestDataset(torch.utils.data.Dataset):
    """测试集数据加载器"""
    
    def __init__(self, image_paths, root_dir, transform=None):
        """
        Args:
            image_paths: 图像路径列表
            root_dir: 图像根目录
            transform: 图像变换
        """
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
            print(f"Error loading image {full_img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path


def predict(model, dataloader, device):
    """
    对数据进行预测
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        predictions: 预测结果字典 {image_path: [5个预测值]}
    """
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for images, img_paths in tqdm(dataloader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.cpu().numpy()
            
            # 保存预测结果
            for i, img_path in enumerate(img_paths):
                predictions[img_path] = outputs[i]
    
    return predictions


def create_submission(test_csv_path, predictions, output_path, target_cols):
    """
    创建提交文件
    
    Args:
        test_csv_path: 测试集 CSV 路径
        predictions: 预测结果字典
        output_path: 输出文件路径
        target_cols: 目标列名列表
    """
    # 读取测试集
    test_df = pd.read_csv(test_csv_path)
    print(f"Test set shape: {test_df.shape}")
    print(f"Test set columns: {test_df.columns.tolist()}")
    
    # 创建提交数据
    submission_data = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Creating submission'):
        sample_id = row['sample_id']
        image_path = row['image_path']
        target_name = row['target_name']
        
        # 获取该图像的预测结果
        if image_path in predictions:
            pred_vector = predictions[image_path]
            
            # 根据 target_name 选择对应的值
            target_idx = target_cols.index(target_name)
            pred_value = pred_vector[target_idx]
        else:
            print(f"Warning: No prediction for {image_path}")
            pred_value = 0.0
        
        submission_data.append({
            'sample_id': sample_id,
            'target': pred_value
        })
    
    # 创建 DataFrame
    submission_df = pd.DataFrame(submission_data)
    
    # 保存
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission file saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")
    print(f"\nFirst few rows:")
    print(submission_df.head(10))
    print(f"\nSubmission statistics:")
    print(submission_df['target'].describe())


def main():
    parser = argparse.ArgumentParser(description='Inference for Biomass Prediction')
    
    # 数据参数
    parser.add_argument('--test_csv', type=str, default='csiro-biomass/test.csv',
                        help='Test CSV file path')
    parser.add_argument('--root_dir', type=str, default='.',
                        help='Root directory for images')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='tf_efficientnet_b0',
                        help='Model architecture')
    parser.add_argument('--weights', type=str, default='output/best_model.pth',
                        help='Path to model weights')
    
    # 推理参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output submission file path')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 目标列名
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    
    # 加载模型
    print("\n=== Loading Model ===")
    model = create_model(
        model_name=args.model,
        pretrained=False,
        device=device
    )
    
    # 加载权重
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded weights from {args.weights}")
    if 'val_rmse' in checkpoint:
        print(f"Model Val RMSE: {checkpoint['val_rmse']:.4f}")
    
    # 读取测试集
    print("\n=== Preparing Test Data ===")
    test_df = pd.read_csv(args.test_csv)
    
    # 获取唯一的图像路径
    unique_image_paths = test_df['image_path'].unique()
    print(f"Unique test images: {len(unique_image_paths)}")
    
    # 创建测试数据集
    test_dataset = TestDataset(
        image_paths=unique_image_paths,
        root_dir=args.root_dir,
        transform=get_transforms(image_size=args.image_size, is_train=False)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 预测
    print("\n=== Running Inference ===")
    predictions = predict(model, test_loader, device)
    print(f"Predictions generated for {len(predictions)} images")
    
    # 创建提交文件
    print("\n=== Creating Submission ===")
    create_submission(
        test_csv_path=args.test_csv,
        predictions=predictions,
        output_path=args.output,
        target_cols=target_cols
    )
    
    print("\n=== Inference Completed ===")


if __name__ == "__main__":
    main()
