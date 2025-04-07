#!/usr/bin/env python
"""
创建模拟数据集脚本，用于测试训练和预测流程
"""

import os
import sys
import pickle
import random
import numpy as np

# 将项目根目录添加到路径中
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

# 氨基酸残基类型列表
restypes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def generate_random_sequence(min_length=50, max_length=200):
    """生成随机蛋白质序列"""
    length = random.randint(min_length, max_length)
    sequence = ''.join(random.choice(restypes) for _ in range(length))
    return sequence

def generate_mock_dataset(num_samples=100, output_path=None):
    """
    生成模拟数据集
    
    参数:
        num_samples: 样本数量
        output_path: 输出文件路径
    
    返回:
        生成的数据集
    """
    # 生成随机序列和标签
    sequences = []
    labels = []
    
    for _ in range(num_samples):
        # 生成随机序列
        seq = generate_random_sequence()
        sequences.append(seq)
        
        # 生成随机标签 (0为有序，1为无序)
        label = [random.randint(0, 1) for _ in range(len(seq))]
        labels.append(label)
    
    # 创建数据集索引
    train_size = int(num_samples * 0.7)
    val_size = int(num_samples * 0.2)
    
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    # 创建数据集字典
    dataset = {
        'seq': sequences,
        'label': labels,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    
    # 保存数据集
    if output_path is None:
        output_path = os.path.join(root_dir, "mock_data.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"模拟数据集已创建，共 {num_samples} 个样本")
    print(f"训练集: {len(train_idx)} 样本")
    print(f"验证集: {len(val_idx)} 样本")
    print(f"测试集: {len(test_idx)} 样本")
    print(f"保存到: {output_path}")
    
    return dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成模拟数据集")
    parser.add_argument("--num_samples", type=int, default=100, help="样本数量")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 生成数据集
    generate_mock_dataset(args.num_samples, args.output) 