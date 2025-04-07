"""
无序蛋白质区域预测模型类文件
"""

import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

# 定义20种标准氨基酸的单字母缩写
restypes = [
    'A', 'R', 'N', 'D', 'C',
    'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
unsure_restype = 'X'  # 表示不确定的氨基酸类型
unknown_restype = 'U'  # 表示未知的氨基酸类型


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    为Transformer模型提供序列位置信息
    """
    def __init__(self, d_model, dropout=0.0, max_len=50000):
        """
        初始化位置编码
        
        参数:
            d_model: 模型维度
            dropout: dropout比率
            max_len: 最大序列长度
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)

        # 使用正弦和余弦函数计算位置编码
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # 将位置编码注册为缓冲区（不参与梯度更新）
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        前向传播，添加位置编码到输入
        
        参数:
            x: 输入张量
        
        返回:
            添加了位置编码的张量
        """
        if len(x.shape) == 3:
            x = x + self.pe[:, : x.size(1)]
        elif len(x.shape) == 4:
            x = x + self.pe[:, :x.size(1), None, :]
        return self.dropout(x)


class DisProtModel(nn.Module):
    """
    无序蛋白质预测模型
    基于Transformer架构实现序列到序列的预测
    """
    def __init__(self, model_config, seq_len=None):
        """
        初始化模型
        
        参数:
            model_config: 模型配置
            seq_len: 序列长度（可选）
        """
        super().__init__()

        self.d_model = model_config.d_model  # 模型维度
        self.n_head = model_config.n_head    # 注意力头数量
        self.n_layer = model_config.n_layer  # Transformer层数

        # 输入映射层
        self.input_layer = nn.Linear(model_config.i_dim, self.d_model)
        # 位置编码，支持长达50000的序列
        self.position_embed = PositionalEncoding(self.d_model, max_len=50000)
        # 输入归一化
        self.input_norm = nn.LayerNorm(self.d_model)
        # 输入Dropout
        self.dropout_in = nn.Dropout(p=0.1)
        # 创建Transformer编码器层
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            activation='gelu',
            batch_first=True)
        # 创建Transformer编码器
        self.transformer = TransformerEncoder(encoder_layer, num_layers=self.n_layer)
        # 输出层，预测每个位置的类别
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.d_model, model_config.o_dim)
        )

    def forward(self, x):
        """
        模型前向传播
        
        参数:
            x: 输入序列张量
        
        返回:
            预测结果张量
        """
        x = self.input_layer(x)           # 输入映射
        x = self.position_embed(x)        # 添加位置编码
        x = self.input_norm(x)            # 归一化
        x = self.dropout_in(x)            # Dropout
        x = self.transformer(x)           # Transformer编码
        x = self.output_layer(x)          # 输出层
        return x 