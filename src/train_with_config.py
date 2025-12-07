"""
训练脚本 - 使用 config.py 配置
实现模型训练和验证流程
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config, cfg
from dataset import BiomassDataset, get_transforms, prepare_data
from model import create_model


def calculate_rmse(predictions, targets):
    """计算 RMSE"""
    mse = np.mean((predictions - targets) ** 2)
    return np.sqrt(mse)


def get_criterion(loss_name):
    """根据配置获取损失函数"""
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'rmse':
        # 自定义 RMSE Loss
        class RMSELoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse = nn.MSELoss()
            def forward(self, pred, target):
                return torch.sqrt(self.mse(pred, target))
        return RMSELoss()
    elif loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'huber':
        return nn.HuberLoss(delta=cfg.HUBER_DELTA)
    elif loss_name == 'smooth_l1':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def get_optimizer(model, optimizer_name):
    """根据配置获取优化器"""
    if optimizer_name == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )
    elif optimizer_name == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )
    elif optimizer_name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name):
    """根据配置获取学习率调度器"""
    if scheduler_name is None:
        return None
    elif scheduler_name == 'cosine':
        params = cfg.SCHEDULER_PARAMS['cosine']
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params['T_max'],
            eta_min=params['eta_min']
        )
    elif scheduler_name == 'step':
        params = cfg.SCHEDULER_PARAMS['step']
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params['step_size'],
            gamma=params['gamma']
        )
    elif scheduler_name == 'plateau':
        params = cfg.SCHEDULER_PARAMS['plateau']
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=params['mode'],
            factor=params['factor'],
            patience=params['patience'],
            min_lr=params['min_lr']
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        if cfg.USE_AMP and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if cfg.GRADIENT_CLIP is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪
            if cfg.GRADIENT_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            
            optimizer.step()
        
        # 记录
        running_loss += loss.item()
        all_preds.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        
        # 更新进度条
        if batch_idx % cfg.LOG_INTERVAL == 0:
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
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE
    ax2.plot(history['train_rmse'], label='Train RMSE', marker='o')
    ax2.plot(history['val_rmse'], label='Val RMSE', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Training and Validation RMSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Training history plot saved to {save_path}")


class EarlyStopping:
    """早停策略"""
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric):
        score = -val_metric  # 因为我们要最小化 RMSE
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'⚠️  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train():
    """主训练函数"""
    # 显示配置
    cfg.display()
    
    # 创建必要的目录
    cfg.create_dirs()
    
    # 设置设备
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # 设置随机种子
    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RANDOM_SEED)
    
    # 准备数据
    print("\n" + "="*60)
    print("📂 Preparing Data")
    print("="*60)
    
    processed_csv = os.path.join(cfg.DATA_ROOT, '../data/processed/train_pivot.csv')
    
    if not os.path.exists(processed_csv):
        print("Processing raw data...")
        prepare_data(cfg.TRAIN_CSV, processed_csv)
    else:
        print(f"Using existing processed data: {processed_csv}")
    
    # 创建数据集
    print("\n📦 Creating Datasets...")
    full_dataset = BiomassDataset(
        csv_path=processed_csv,
        root_dir='csiro-biomass',
        transform=get_transforms(image_size=cfg.IMAGE_SIZE, is_train=True),
        is_train=True
    )
    
    # 划分训练集和验证集
    train_size = int((1 - cfg.TRAIN_VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
    )
    
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Val size: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    # 创建模型
    print("\n" + "="*60)
    print("🏗️  Creating Model")
    print("="*60)
    model = create_model(
        model_name=cfg.MODEL_NAME,
        pretrained=cfg.PRETRAINED,
        device=device
    )
    
    # 定义损失函数、优化器和调度器
    criterion = get_criterion(cfg.LOSS_FUNCTION)
    optimizer = get_optimizer(model, cfg.OPTIMIZER)
    scheduler = get_scheduler(optimizer, cfg.SCHEDULER)
    
    print(f"\n📊 Training Configuration:")
    print(f"  Loss: {cfg.LOSS_FUNCTION}")
    print(f"  Optimizer: {cfg.OPTIMIZER}")
    print(f"  Scheduler: {cfg.SCHEDULER}")
    print(f"  Learning Rate: {cfg.LEARNING_RATE}")
    print(f"  Gradient Clip: {cfg.GRADIENT_CLIP}")
    print(f"  Mixed Precision: {cfg.USE_AMP}")
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if cfg.USE_AMP else None
    
    # 早停
    early_stopping = EarlyStopping(
        patience=cfg.EARLY_STOPPING_PATIENCE,
        verbose=True
    ) if cfg.EARLY_STOPPING else None
    
    # 训练循环
    print("\n" + "="*60)
    print("🚀 Training Started")
    print("="*60)
    
    best_val_rmse = float('inf')
    history = {
        'train_loss': [],
        'train_rmse': [],
        'val_loss': [],
        'val_rmse': []
    }
    
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.EPOCHS}")
        print(f"{'='*60}")
        
        # 训练
        train_loss, train_rmse = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
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
        print(f"\n📈 Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val RMSE:   {val_rmse:.4f}")
        
        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'val_loss': val_loss,
                'config': {
                    'model_name': cfg.MODEL_NAME,
                    'image_size': cfg.IMAGE_SIZE,
                    'num_classes': cfg.NUM_CLASSES
                }
            }, cfg.MODEL_SAVE_PATH)
            print(f"  ✅ Best model saved (RMSE: {best_val_rmse:.4f})")
        
        # 早停检查
        if early_stopping is not None:
            early_stopping(val_rmse)
            if early_stopping.early_stop:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
                break
    
    # 绘制训练历史
    plot_path = os.path.join(cfg.OUTPUT_DIR, 'training_history.png')
    plot_history(history, plot_path)
    
    # 训练完成
    print("\n" + "="*60)
    print("✅ Training Completed")
    print("="*60)
    print(f"Best Val RMSE: {best_val_rmse:.4f}")
    print(f"Model saved to: {cfg.MODEL_SAVE_PATH}")
    print(f"Output directory: {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    train()
