"""
无序蛋白质预测模型训练主脚本
"""

import os
import argparse
import datetime
import torch
from torch.utils.data import DataLoader

from src.models.transformer import DisProtTransformer
from src.data.dataset import make_dataset
from src.train.trainer import Trainer
from src.utils.config import load_config

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="无序蛋白质预测模型训练")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--use_timestamp", action="store_true", help="为输出目录添加时间戳")
    parser.add_argument("--num_gpus", type=int, default=1, help="使用的GPU数量")
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    if args.use_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, timestamp)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 准备数据集
    print("加载数据集...")
    train_dataset, val_dataset, test_dataset = make_dataset(config=config)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.dataloader.batch_size,
        shuffle=config.train.dataloader.shuffle,
        num_workers=config.train.dataloader.num_workers,
        drop_last=config.train.dataloader.drop_last
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.train.dataloader.batch_size,
        shuffle=False,
        num_workers=config.train.dataloader.num_workers,
        drop_last=False
    )
    
    # 创建模型
    print("初始化模型...")
    model = DisProtTransformer(config)
    model.to(device)
    
    # 多GPU训练设置
    if torch.cuda.device_count() > 1 and args.num_gpus > 1:
        print(f"使用 {min(torch.cuda.device_count(), args.num_gpus)} 个GPU进行训练")
        model = torch.nn.DataParallel(model, device_ids=list(range(min(torch.cuda.device_count(), args.num_gpus))))
    
    # 创建训练器
    trainer = Trainer(model, config, device)
    
    # 开始训练
    print("开始训练...")
    best_model_path = trainer.train(train_dataloader, val_dataloader, output_dir)
    
    print(f"训练完成! 最佳模型保存在: {best_model_path}")

if __name__ == "__main__":
    main() 