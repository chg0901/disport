"""
无序蛋白质预测模型预测脚本
"""

import os
import glob
import argparse
import torch
from torch.utils.data import DataLoader

from src.models.transformer import DisProtTransformer
from src.data.dataset import make_dataset
from src.utils.config import load_config
from src.utils.metrics import metric_fn

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="无序蛋白质预测模型预测")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--model_path", type=str, default=None, help="模型文件路径，留空自动查找最佳模型")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录路径")
    return parser.parse_args()

def find_best_model(output_dir):
    """
    在输出目录中查找最佳模型
    
    参数:
        output_dir: 输出目录路径
        
    返回:
        最佳模型文件路径
    """
    # 首先检查是否存在标准命名的最佳模型
    standard_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(standard_path):
        return standard_path
    
    # 检查顶层目录中带有F1分数的模型
    top_models = glob.glob(os.path.join(output_dir, "best_model_f1_*.pth"))
    if top_models:
        # 按文件修改时间排序，返回最新的
        return sorted(top_models, key=os.path.getmtime)[-1]
    
    # 检查时间戳子目录中的模型
    timestamp_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    for ts_dir in sorted(timestamp_dirs, reverse=True):  # 从最新的开始找
        ts_dir_path = os.path.join(output_dir, ts_dir)
        
        # 检查子目录中带有F1分数的模型
        sub_models = glob.glob(os.path.join(ts_dir_path, "best_model_f1_*.pth"))
        if sub_models:
            return sorted(sub_models, key=os.path.getmtime)[-1]
        
        # 检查子目录中标准命名的模型
        sub_standard = os.path.join(ts_dir_path, "best_model.pth")
        if os.path.exists(sub_standard):
            return sub_standard
    
    raise FileNotFoundError(f"未找到任何模型文件，请检查 {output_dir} 目录")

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 准备数据集
    print("加载数据集...")
    _, _, test_dataset = make_dataset(config=config)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.train.dataloader.batch_size,
        shuffle=False,
        num_workers=config.train.dataloader.num_workers,
        drop_last=False
    )
    
    # 创建模型
    print("初始化模型...")
    model = DisProtTransformer(config)
    
    # 加载模型权重
    model_path = args.model_path if args.model_path else find_best_model(args.output_dir)
    print(f"加载模型权重: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # 评估模型
    print("开始评估...")
    metric = 0.0
    with torch.no_grad():
        for sequence, label in test_dataloader:
            sequence = sequence.to(device)
            label = label.to(device)
            
            pred = model(sequence)
            metric += metric_fn(pred, label)
    
    test_f1 = metric / len(test_dataloader)
    print(f"测试集F1分数: {test_f1:.4f}")
    
    return test_f1

if __name__ == "__main__":
    main() 