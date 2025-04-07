"""
Transformer模型模块，用于无序蛋白质预测
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码层
        
        参数:
            d_model: 模型的维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为一个缓冲区（不会被当作模型参数）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码到输入中
        
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        返回:
            添加了位置编码的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class Transformer(nn.Module):
    """无序蛋白质预测的Transformer模型"""
    
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, in_dim, out_dim):
        """
        初始化Transformer模型
        
        参数:
            d_model: 模型隐藏维度
            nhead: 注意力头数量
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数（不使用解码器时为0）
            dim_feedforward: 前馈网络隐藏层维度
            dropout: Dropout概率
            in_dim: 输入维度
            out_dim: 输出维度（类别数）
        """
        super(Transformer, self).__init__()
        
        # 输入嵌入层
        self.embedding = nn.Linear(in_dim, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, out_dim)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, in_dim]
            
        返回:
            每个位置的预测结果 [batch_size, seq_len, out_dim]
        """
        # 嵌入并添加位置编码
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        # 输出层
        logits = self.output_layer(x)
        
        return logits

class DisProtTransformer(nn.Module):
    """无序蛋白质预测的Transformer模型"""
    
    def __init__(self, config):
        """
        初始化Transformer模型
        
        参数:
            config: 配置对象，包含模型配置信息
        """
        super(DisProtTransformer, self).__init__()
        
        # 从配置中获取模型参数
        d_model = config.model.d_model
        n_head = config.model.n_head
        n_layer = config.model.n_layer
        i_dim = config.model.i_dim
        o_dim = config.model.o_dim
        
        # 输入嵌入层
        self.embedding = nn.Linear(i_dim, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, o_dim)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, i_dim]
            
        返回:
            每个位置的预测结果 [batch_size, seq_len, o_dim]
        """
        # 嵌入并添加位置编码
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        # 输出层
        logits = self.output_layer(x)
        
        return logits 