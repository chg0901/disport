"""
配置工具模块，用于加载和解析配置文件
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ModelConfig:
    """模型配置"""
    d_model: int = 512
    n_head: int = 8
    n_layer: int = 6
    i_dim: int = 21  # 输入维度（氨基酸编码）
    o_dim: int = 2   # 输出维度（无序/有序）

@dataclass
class DataConfig:
    """数据配置"""
    data_path: str = "WSAA_data_public.pkl"
    max_seq_len: Optional[int] = None
    train_rate: float = 0.7  # 训练集比例（若数据集未预分割）
    valid_rate: float = 0.2  # 验证集比例（若数据集未预分割）

@dataclass
class DataloaderConfig:
    """数据加载器配置"""
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 4
    drop_last: bool = False

@dataclass
class OptimizerConfig:
    """优化器配置"""
    lr: float = 1e-4
    weight_decay: float = 1e-4

@dataclass
class TrainConfig:
    """训练配置"""
    epochs: int = 50
    save_interval: int = 1
    eval_interval: int = 1
    early_stop_patience: int = 10  # 早停轮数
    checkpoint_dir: str = "checkpoints"  # 检查点保存目录
    use_gpu: bool = True  # 是否使用GPU
    gpu: int = 0  # 使用的GPU编号，-1表示使用CPU
    gpu_ids: List[int] = field(default_factory=lambda: [0])  # 使用的GPU ID列表
    seed: int = 42  # 随机种子
    use_swanlab: bool = False  # 是否使用SwanLab
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

@dataclass
class SwanlabConfig:
    """SwanLab配置"""
    enabled: bool = True
    experiment_name: str = "disport"
    save_path: str = "swanlab_logs"
    monitoring_interval: int = 5

@dataclass
class PredictConfig:
    """预测配置"""
    output_dir: str = "predictions"  # 预测结果输出目录
    visualize: bool = True  # 是否可视化预测结果

@dataclass
class Config:
    """全局配置"""
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    predict: PredictConfig
    swanlab: Optional[SwanlabConfig] = field(default_factory=SwanlabConfig)

def load_config(config_path: str) -> Config:
    """
    加载YAML配置文件并解析为配置对象
    
    参数:
        config_path: 配置文件路径
        
    返回:
        配置对象
    """
    # 检查文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    
    # 加载YAML文件
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 解析配置字典并创建配置对象
    model_config = ModelConfig(**config_dict.get('model', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    
    # 创建训练配置
    train_dict = config_dict.get('train', {})
    dataloader_config = DataloaderConfig(**train_dict.get('dataloader', {}))
    optimizer_config = OptimizerConfig(**train_dict.get('optimizer', {}))
    
    # 移除已处理的子配置项
    if 'dataloader' in train_dict:
        train_dict.pop('dataloader')
    if 'optimizer' in train_dict:
        train_dict.pop('optimizer')
    
    train_config = TrainConfig(**train_dict)
    train_config.dataloader = dataloader_config
    train_config.optimizer = optimizer_config
    
    # 处理预测配置
    predict_config = None
    if 'predict' in config_dict:
        predict_config = PredictConfig(**config_dict.get('predict', {}))
    else:
        predict_config = PredictConfig()
    
    # 处理SwanLab配置
    swanlab_config = None
    if 'swanlab' in config_dict:
        swanlab_config = SwanlabConfig(**config_dict.get('swanlab', {}))
    
    # 创建并返回全局配置对象
    return Config(
        model=model_config,
        data=data_config,
        train=train_config,
        predict=predict_config,
        swanlab=swanlab_config
    ) 