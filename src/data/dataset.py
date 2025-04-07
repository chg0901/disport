"""
数据集模块，用于加载和处理无序蛋白质预测的数据集
"""

import os
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

# 氨基酸残基类型列表
restypes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

class DisProtDataset(Dataset):
    """无序蛋白质数据集类"""
    
    def __init__(self, sequences, labels):
        """
        初始化数据集
        
        参数:
            sequences: 蛋白质序列列表
            labels: 对应的标签列表
        """
        self.sequences = sequences
        self.labels = labels
        
        # 为每个氨基酸创建one-hot映射
        self.residue_mapping = {'X': 20}
        self.residue_mapping.update(dict(zip(restypes, range(len(restypes)))))
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # 创建输入序列的one-hot编码
        seq_tensor = torch.zeros(len(seq), len(self.residue_mapping))
        for i, char in enumerate(seq):
            if char not in restypes:
                char = 'X'
            seq_tensor[i][self.residue_mapping[char]] = 1
        
        # 将标签转换为长整型，用于CrossEntropyLoss
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return seq_tensor, label_tensor


def collate_fn(batch):
    """
    将不同长度的序列批处理在一起
    
    参数:
        batch: 一批样本，每个样本包含序列张量和标签张量
        
    返回:
        批处理后的序列张量和标签张量
    """
    # 提取序列和标签
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 获取最大序列长度
    max_len = max([seq.size(0) for seq in sequences])
    
    # 填充序列
    padded_sequences = []
    padded_labels = []
    for seq, label in zip(sequences, labels):
        # 序列填充
        seq_len = seq.size(0)
        if seq_len < max_len:
            padding = torch.zeros(max_len - seq_len, seq.size(1))
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
        
        # 标签填充
        label_len = label.size(0)
        if label_len < max_len:
            padding = torch.zeros(max_len - label_len, dtype=torch.long)
            padded_label = torch.cat([label, padding], dim=0)
        else:
            padded_label = label
        padded_labels.append(padded_label)
    
    # 堆叠张量
    padded_sequences = torch.stack(padded_sequences, dim=0)
    padded_labels = torch.stack(padded_labels, dim=0)
    
    return padded_sequences, padded_labels


def make_dataset(config):
    """
    根据配置创建训练、验证和测试数据集
    
    参数:
        config: 配置对象
        
    返回:
        训练、验证和测试数据集的元组
    """
    # 获取数据路径（支持相对路径和绝对路径）
    data_path = config.data.data_path
    if not os.path.isabs(data_path):
        # 如果是相对路径，则相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(project_root, data_path)
    
    print(f"加载数据文件: {data_path}")
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件: {data_path}")
    
    # 加载数据集
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 检查数据结构，判断数据集格式
    if isinstance(data, dict) and 'train' in data and 'val' in data and 'test' in data:
        # 新格式：包含预先分割的train/val/test子集
        print("使用数据集预定义的训练/验证/测试分割")
        
        train_dataset = DisProtDataset(
            data['train']['sequences'],
            data['train']['labels']
        )
        
        val_dataset = DisProtDataset(
            data['val']['sequences'],
            data['val']['labels']
        )
        
        test_dataset = DisProtDataset(
            data['test']['sequences'],
            data['test']['labels']
        )
    elif isinstance(data, dict) and 'seq' in data and 'label' in data:
        # 旧格式：使用索引进行分割
        print("使用索引分割数据集")
        has_predefined_split = 'train_idx' in data and 'val_idx' in data and 'test_idx' in data
        
        if has_predefined_split:
            print("使用数据集预定义的训练/验证/测试分割")
            # 使用预定义的分割
            train_idx = data['train_idx']
            val_idx = data['val_idx']
            test_idx = data['test_idx']
        else:
            print("根据配置进行数据分割")
            # 根据配置进行分割
            total_count = len(data['seq'])
            train_count = int(total_count * config.data.train_rate)
            val_count = int(total_count * config.data.valid_rate)
            
            # 创建索引
            indices = list(range(total_count))
            train_idx = indices[:train_count]
            val_idx = indices[train_count:train_count+val_count]
            test_idx = indices[train_count+val_count:]
        
        train_dataset = DisProtDataset(
            [data['seq'][i] for i in train_idx],
            [data['label'][i] for i in train_idx]
        )
        
        val_dataset = DisProtDataset(
            [data['seq'][i] for i in val_idx],
            [data['label'][i] for i in val_idx]
        )
        
        test_dataset = DisProtDataset(
            [data['seq'][i] for i in test_idx],
            [data['label'][i] for i in test_idx]
        )
    else:
        raise ValueError("不支持的数据集格式")
    
    print(f"数据集加载完成 - 训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本, 测试集: {len(test_dataset)} 样本")
    
    return train_dataset, val_dataset, test_dataset 