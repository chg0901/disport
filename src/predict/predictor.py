"""
预测器模块，用于使用训练好的无序蛋白质预测模型进行预测
"""

import os
import torch
import numpy as np
from tqdm import tqdm

from src.models.model import restypes

class Predictor:
    """
    预测器类，封装模型预测逻辑
    """
    
    def __init__(self, model, device=None):
        """
        初始化预测器
        
        参数:
            model: 训练好的模型实例
            device: 计算设备，如果为None则自动选择
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 构建氨基酸到索引的映射
        self.residue_mapping = {'X': 20}
        self.residue_mapping.update(dict(zip(restypes, range(len(restypes)))))
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, model_class, device=None):
        """
        从检查点文件加载预测器
        
        参数:
            checkpoint_path: 检查点文件路径
            model_class: 模型类
            device: 计算设备
            
        返回:
            预测器实例
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # 从检查点获取配置
        config = checkpoint.get('config')
        
        # 创建模型实例
        model = model_class(
            seq_len=None,  # 预测时序列长度可变
            embedding_dim=config.model.embed_dim,
            ff_dim=config.model.ff_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        )
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建预测器
        return cls(model, device)
    
    def preprocess_sequence(self, sequence):
        """
        预处理输入序列
        
        参数:
            sequence: 氨基酸序列字符串
            
        返回:
            预处理后的序列张量
        """
        # 创建one-hot编码的序列表示
        sequence_tensor = torch.zeros(len(sequence), len(self.residue_mapping))
        for i, c in enumerate(sequence):
            if c not in restypes:
                c = 'X'  # 如果不是标准氨基酸，则标记为'X'
            sequence_tensor[i][self.residue_mapping[c]] = 1
        
        return sequence_tensor.unsqueeze(0)  # 添加批次维度
    
    def predict(self, sequence, threshold=0.5):
        """
        预测单个序列的无序区域
        
        参数:
            sequence: 氨基酸序列字符串
            threshold: 预测阈值
            
        返回:
            预测结果字典，包含原始概率和二分类结果
        """
        # 预处理序列
        sequence_tensor = self.preprocess_sequence(sequence)
        sequence_tensor = sequence_tensor.to(self.device)
        
        # 进行预测
        with torch.no_grad():
            output = self.model(sequence_tensor)
            probabilities = torch.sigmoid(output).squeeze()
        
        # 转换为二分类结果
        binary_prediction = (probabilities >= threshold).int()
        
        # 转换为numpy数组并返回结果
        probs_np = probabilities.cpu().numpy()
        binary_np = binary_prediction.cpu().numpy()
        
        return {
            'sequence': sequence,
            'probabilities': probs_np,
            'binary_prediction': binary_np,
            'disorder_regions': self._get_disorder_regions(binary_np)
        }
    
    def _get_disorder_regions(self, binary_prediction):
        """
        从二分类预测结果中提取无序区域
        
        参数:
            binary_prediction: 二分类预测结果数组
            
        返回:
            无序区域列表，每个元素为(起始位置, 结束位置)
        """
        regions = []
        start = None
        
        for i, pred in enumerate(binary_prediction):
            if pred == 1 and start is None:
                start = i
            elif pred == 0 and start is not None:
                regions.append((start, i - 1))
                start = None
        
        # 处理序列末尾的无序区域
        if start is not None:
            regions.append((start, len(binary_prediction) - 1))
        
        return regions
    
    def batch_predict(self, sequences, threshold=0.5, batch_size=8):
        """
        批量预测多个序列
        
        参数:
            sequences: 序列列表
            threshold: 预测阈值
            batch_size: 批处理大小
            
        返回:
            预测结果列表
        """
        results = []
        
        # 将序列分批处理
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="预测中"):
            batch_sequences = sequences[i * batch_size:(i + 1) * batch_size]
            batch_results = [self.predict(seq, threshold) for seq in batch_sequences]
            results.extend(batch_results)
        
        return results
    
    def save_predictions(self, predictions, output_dir):
        """
        保存预测结果到文件
        
        参数:
            predictions: 预测结果列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存概率文件
        with open(os.path.join(output_dir, 'predictions.txt'), 'w') as f:
            for pred in predictions:
                seq = pred['sequence']
                probs = pred['probabilities']
                f.write(f">{seq[:10]}...\n")
                for i, p in enumerate(probs):
                    f.write(f"{i+1}\t{seq[i]}\t{p:.4f}\n")
                f.write("\n")
        
        # 保存无序区域文件
        with open(os.path.join(output_dir, 'disorder_regions.txt'), 'w') as f:
            for pred in predictions:
                seq = pred['sequence']
                regions = pred['disorder_regions']
                f.write(f">{seq[:10]}...\n")
                for start, end in regions:
                    f.write(f"{start+1}-{end+1}\n")
                f.write("\n") 