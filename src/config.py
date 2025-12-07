"""
配置文件 - 集中管理所有训练和推理参数
"""

import os


class Config:
    """训练和推理配置类"""
    
    # ==================== 路径配置 ====================
    # 数据路径
    DATA_ROOT = '/root/autodl-tmp/CSIRO/csiro-biomass'
    TRAIN_CSV = os.path.join(DATA_ROOT, 'train.csv')
    TRAIN_IMG_ROOT = os.path.join(DATA_ROOT, 'train')
    TEST_CSV = os.path.join(DATA_ROOT, 'test.csv')
    TEST_IMG_ROOT = os.path.join(DATA_ROOT, 'test')
    
    # 输出路径
    OUTPUT_DIR = '/root/autodl-tmp/CSIRO/output'
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    # ==================== 模型配置 ====================
    # 模型架构
    MODEL_NAME = 'tf_efficientnet_b0'  # timm 模型名称
    PRETRAINED = True  # 是否使用预训练权重
    NUM_CLASSES = 5  # 输出类别数（5个生物量指标）
    
    # 可选的其他模型（性能从低到高）
    # MODEL_NAME = 'tf_efficientnet_b1'
    # MODEL_NAME = 'tf_efficientnet_b3'
    # MODEL_NAME = 'convnext_tiny'
    # MODEL_NAME = 'convnext_small'
    # MODEL_NAME = 'swin_tiny_patch4_window7_224'
    # MODEL_NAME = 'swin_small_patch4_window7_224'
    
    # ==================== 训练配置 ====================
    # 基础训练参数
    EPOCHS = 50
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = 'cuda'  # 'cuda' 或 'cpu'
    
    # 优化器参数
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = 'adamw'  # 'adam', 'adamw', 'sgd'
    
    # 学习率调度器
    SCHEDULER = 'cosine'  # 'cosine', 'step', 'plateau', None
    SCHEDULER_PARAMS = {
        'cosine': {
            'T_max': 50,  # 余弦退火周期
            'eta_min': 1e-6  # 最小学习率
        },
        'step': {
            'step_size': 10,  # 每隔多少 epoch 降低学习率
            'gamma': 0.1  # 学习率衰减因子
        },
        'plateau': {
            'mode': 'min',
            'factor': 0.5,  # 学习率衰减因子
            'patience': 5,  # 容忍多少 epoch 不改善
            'min_lr': 1e-6
        }
    }
    
    # 早停策略
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10  # 容忍多少 epoch 验证集不改善
    
    # ==================== 数据增强配置 ====================
    # 图像尺寸
    IMAGE_SIZE = 224
    
    # 训练集数据增强
    TRAIN_AUGMENTATION = {
        'resize': IMAGE_SIZE,
        'random_crop': True,
        'horizontal_flip': True,
        'vertical_flip': False,
        'rotation': 15,  # 随机旋转角度
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        },
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    
    # 验证集/测试集数据增强
    VAL_AUGMENTATION = {
        'resize': IMAGE_SIZE,
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    
    # ==================== 数据集配置 ====================
    # 数据集划分
    TRAIN_VAL_SPLIT = 0.2  # 验证集比例
    RANDOM_SEED = 42
    
    # 目标列名
    TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    
    # ==================== 损失函数配置 ====================
    LOSS_FUNCTION = 'mse'  # 'mse', 'rmse', 'mae', 'huber', 'smooth_l1'
    
    # Huber Loss 参数
    HUBER_DELTA = 1.0
    
    # ==================== 混合精度训练 ====================
    USE_AMP = True  # 是否使用自动混合精度训练（加速训练）
    
    # ==================== 梯度裁剪 ====================
    GRADIENT_CLIP = 1.0  # 梯度裁剪阈值，None 表示不裁剪
    
    # ==================== 日志和保存配置 ====================
    # 日志频率
    LOG_INTERVAL = 10  # 每隔多少个 batch 打印一次日志
    SAVE_BEST_ONLY = True  # 是否只保存最佳模型
    
    # TensorBoard
    USE_TENSORBOARD = False
    TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, 'tensorboard')
    
    # ==================== 推理配置 ====================
    # 测试时批量大小
    TEST_BATCH_SIZE = 64
    
    # TTA (Test Time Augmentation)
    USE_TTA = False
    TTA_TRANSFORMS = ['original', 'hflip', 'vflip']
    
    @classmethod
    def display(cls):
        """打印所有配置"""
        print("=" * 60)
        print("📋 Configuration Settings")
        print("=" * 60)
        
        sections = {
            'Path': ['DATA_ROOT', 'OUTPUT_DIR', 'MODEL_SAVE_PATH'],
            'Model': ['MODEL_NAME', 'PRETRAINED', 'NUM_CLASSES'],
            'Training': ['EPOCHS', 'BATCH_SIZE', 'LEARNING_RATE', 'OPTIMIZER', 'SCHEDULER'],
            'Data': ['IMAGE_SIZE', 'TRAIN_VAL_SPLIT', 'RANDOM_SEED'],
            'Loss': ['LOSS_FUNCTION'],
            'Advanced': ['USE_AMP', 'GRADIENT_CLIP', 'EARLY_STOPPING']
        }
        
        for section, keys in sections.items():
            print(f"\n{section} Configuration:")
            for key in keys:
                value = getattr(cls, key, 'N/A')
                print(f"  {key}: {value}")
        
        print("=" * 60)
    
    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        if cls.USE_TENSORBOARD:
            os.makedirs(cls.TENSORBOARD_DIR, exist_ok=True)
        print(f"✅ Directories created: {cls.OUTPUT_DIR}")


# 创建全局配置实例
cfg = Config()


if __name__ == "__main__":
    # 测试配置
    Config.display()
    Config.create_dirs()
