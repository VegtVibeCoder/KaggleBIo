"""
训练脚本
实现模型训练和验证流程
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import BiomassDataset, get_transforms, prepare_data
from model import create_model


def calculate_rmse(predictions, targets):
    """计算 RMSE"""
    mse = np.mean((predictions - targets) ** 2)
    return np.sqrt(mse)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 记录
        running_loss += loss.item()
        all_preds.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    epoch_rmse = calculate_rmse(all_preds, all_targets)
    
    return epoch_loss, epoch_rmse


def validate_epoch(model, dataloader, criterion, device, epoch):
    """验证一个 epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Valid]')
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # 记录
            running_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    epoch_rmse = calculate_rmse(all_preds, all_targets)
    
    return epoch_loss, epoch_rmse


def plot_history(history, save_path):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # RMSE
    ax2.plot(history['train_rmse'], label='Train RMSE')
    ax2.plot(history['val_rmse'], label='Val RMSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Training and Validation RMSE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")


def train(args):
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备数据
    print("\n=== Preparing Data ===")
    processed_csv = os.path.join(args.data_dir, 'processed/train_pivot.csv')
    
    if not os.path.exists(processed_csv):
        print("Processing raw data...")
        raw_csv = os.path.join(args.data_dir, '../csiro-biomass/train.csv')
        prepare_data(raw_csv, processed_csv)
    else:
        print(f"Using existing processed data: {processed_csv}")
    
    # 创建数据集
    print("\n=== Creating Datasets ===")
    full_dataset = BiomassDataset(
        csv_path=processed_csv,
        root_dir='csiro-biomass',
        transform=get_transforms(image_size=args.image_size, is_train=True),
        is_train=True
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    print("\n=== Creating Model ===")
    model = create_model(
        model_name=args.model,
        pretrained=args.pretrained,
        device=device
    )
    
    # 定义损失函数和优化器
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'l1':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # 训练循环
    print("\n=== Training ===")
    best_val_rmse = float('inf')
    history = {
        'train_loss': [],
        'train_rmse': [],
        'val_loss': [],
        'val_rmse': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 训练
        train_loss, train_rmse = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_rmse = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_rmse'].append(train_rmse)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"✓ Best model saved (RMSE: {best_val_rmse:.4f})")
    
    # 保存最后的模型
    last_model_path = os.path.join(args.output_dir, 'last_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_rmse': val_rmse,
        'val_loss': val_loss,
    }, last_model_path)
    
    # 绘制训练历史
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_history(history, plot_path)
    
    print("\n=== Training Completed ===")
    print(f"Best Val RMSE: {best_val_rmse:.4f}")
    print(f"Models saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Biomass Prediction Model')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='tf_efficientnet_b0',
                        choices=['tf_efficientnet_b0', 'resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'l1'],
                        help='Loss function')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 开始训练
    train(args)


if __name__ == "__main__":
    main()
