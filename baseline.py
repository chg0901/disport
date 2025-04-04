import argparse
import math
import pickle
import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import swanlab  # 导入SwanLab

# 定义20种标准氨基酸的单字母缩写
restypes = [
    'A', 'R', 'N', 'D', 'C',
    'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
unsure_restype = 'X'  # 表示不确定的氨基酸类型
unknown_restype = 'U'  # 表示未知的氨基酸类型

def make_dataset(data_config, train_rate=0.7, valid_rate=0.2):
    """
    根据配置加载数据并分割为训练集、验证集和测试集
    
    参数:
        data_config: 数据配置
        train_rate: 训练集比例
        valid_rate: 验证集比例
    
    返回:
        train_dataset, valid_dataset, test_dataset: 三个数据集对象
    """
    data_path = data_config.data_path
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 计算数据分割点
    total_number = len(data)
    train_sep = int(total_number * train_rate)
    valid_sep = int(total_number * (train_rate + valid_rate))

    # 分割数据
    train_data_dicts = data[:train_sep]
    valid_data_dicts = data[train_sep:valid_sep]
    test_data_dicts = data[valid_sep:]

    # 创建数据集对象
    train_dataset = DisProtDataset(train_data_dicts)
    valid_dataset = DisProtDataset(valid_data_dicts)
    test_dataset = DisProtDataset(test_data_dicts)

    return train_dataset, valid_dataset, test_dataset


class DisProtDataset(Dataset):
    """
    无序蛋白质预测数据集类
    用于加载和预处理蛋白质序列及其对应的标签
    """
    def __init__(self, dict_data):
        """
        初始化数据集
        
        参数:
            dict_data: 包含序列和标签的字典列表
        """
        sequences = [d['sequence'] for d in dict_data]
        labels = [d['label'] for d in dict_data]
        assert len(sequences) == len(labels)

        self.sequences = sequences
        self.labels = labels
        # 构建氨基酸到索引的映射，'X'映射到20
        self.residue_mapping = {'X':20}
        self.residue_mapping.update(dict(zip(restypes, range(len(restypes)))))

    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        获取一个数据样本
        
        参数:
            idx: 样本索引
        
        返回:
            sequence: one-hot编码的氨基酸序列
            label: 对应的标签序列
        """
        # 创建one-hot编码的序列表示
        sequence = torch.zeros(len(self.sequences[idx]), len(self.residue_mapping))
        for i, c in enumerate(self.sequences[idx]):
            if c not in restypes:
                c = 'X'  # 如果不是标准氨基酸，则标记为'X'
            sequence[i][self.residue_mapping[c]] = 1

        # 转换标签为张量
        label = torch.tensor([int(c) for c in self.labels[idx]], dtype=torch.long)
        return sequence, label


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    为Transformer模型提供序列位置信息
    """
    def __init__(self, d_model, dropout=0.0, max_len=40):
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
    def __init__(self, model_config):
        """
        初始化模型
        
        参数:
            model_config: 模型配置
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


def metric_fn(pred, gt):
    """
    计算评估指标（F1分数）
    
    参数:
        pred: 预测结果
        gt: 真实标签
    
    返回:
        F1分数
    """
    pred = pred.detach().cpu()
    gt = gt.detach().cpu()
    pred_labels = torch.argmax(pred, dim=-1).view(-1)  # 获取预测类别
    gt_labels = gt.view(-1)  # 展平真实标签
    score = f1_score(y_true=gt_labels, y_pred=pred_labels, average='micro')  # 计算微平均F1分数
    return score


def main():
    """主函数，处理命令行参数并运行训练流程"""
    # 选择设备（GPU或CPU）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 解析命令行参数
    parser = argparse.ArgumentParser('IDRs prediction')
    parser.add_argument('--config_path', default='./config.yaml')
    parser.add_argument('--output_dir', default='./outputs', help='输出目录')
    parser.add_argument('--use_swanlab', action='store_true', help='是否使用SwanLab进行可视化')
    parser.add_argument('--no_timestamp', action='store_true', help='不添加时间戳到输出目录')
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config_path)  # 加载配置文件
    
    # 添加时间戳到输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if not args.no_timestamp:
        output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 同时确保顶层输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化SwanLab（如果需要）
    if args.use_swanlab:
        # 将配置转换为字典以便SwanLab记录
        config_dict = OmegaConf.to_container(config, resolve=True)
        # 添加运行时间戳和输出目录到配置
        config_dict['run_timestamp'] = timestamp
        config_dict['output_directory'] = output_dir
        swanlab.init(
            project="disprot-prediction",
            config=config_dict,
            description=f"无序蛋白质区域预测模型训练 - {timestamp}",
        )

    # 准备数据集和数据加载器
    train_dataset, valid_dataset, test_dataset = make_dataset(config.data)
    train_dataloader = DataLoader(dataset=train_dataset, **config.train.dataloader)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)

    # 初始化模型
    model = DisProtModel(config.model)
    model = model.to(device)

    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=config.train.optimizer.lr,
                                weight_decay=config.train.optimizer.weight_decay)
    # 设置损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 初始评估
    model.eval()
    metric = 0.
    with torch.no_grad():
        for sequence, label in valid_dataloader:
            sequence = sequence.to(device)
            label = label.to(device)
            pred = model(sequence)
            metric += metric_fn(pred, label)
    init_f1 = metric / len(valid_dataloader)
    print("init f1_score:", init_f1)
    
    # 记录初始F1分数到SwanLab
    if args.use_swanlab:
        swanlab.log({"val_f1": init_f1})

    # 训练循环
    best_val_f1 = 0.0
    for epoch in range(config.train.epochs):
        # 训练阶段
        progress_bar = tqdm(
            train_dataloader,
            initial=0,
            desc=f"epoch:{epoch:03d}",
        )
        model.train()
        total_loss = 0.
        for sequence, label in progress_bar:
            sequence = sequence.to(device)
            label = label.to(device)

            # 前向传播
            pred = model(sequence)
            # 计算损失（需要调整维度顺序以匹配CrossEntropyLoss的要求）
            loss = loss_fn(pred.permute(0, 2, 1), label)
            progress_bar.set_postfix(loss=loss.item())
            total_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_dataloader)

        # 验证阶段
        model.eval()
        metric = 0.
        with torch.no_grad():
            for sequence, label in valid_dataloader:
                sequence = sequence.to(device)
                label = label.to(device)
                pred = model(sequence)
                metric += metric_fn(pred, label)
        val_f1 = metric / len(valid_dataloader)
        print(f"avg_training_loss: {avg_loss}, f1_score: {val_f1}")
        
        # 记录训练指标到SwanLab
        if args.use_swanlab:
            swanlab.log({
                "train_loss": avg_loss,
                "val_f1": val_f1,
                "epoch": epoch
            })
        
        # 保存模型
        model_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        
        # 保存最佳模型
        if epoch == 0 or val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # 添加F1分数和时间戳到最佳模型文件名
            best_model_path = os.path.join(output_dir, f"best_model_f1_{best_val_f1:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)
            # 复制最佳模型到顶层输出目录，并在文件名中保留F1分数和时间戳
            top_best_model_path = os.path.join(args.output_dir, f"best_model_f1_{best_val_f1:.4f}_{timestamp}.pth")
            import shutil
            shutil.copy2(best_model_path, top_best_model_path)
            # 同时创建一个固定名称的副本，方便预测脚本使用
            standard_best_model_path = os.path.join(args.output_dir, "best_model.pth")
            shutil.copy2(best_model_path, standard_best_model_path)
            
            if args.use_swanlab:
                swanlab.log({"best_val_f1": best_val_f1})
    
    # 训练结束，记录最终结果
    if args.use_swanlab:
        swanlab.log({
            "final_val_f1": val_f1,
            "best_val_f1": best_val_f1,
            "total_epochs": config.train.epochs,
            "run_timestamp": timestamp,
            "output_directory": output_dir
        })
        # 保存模型文件到SwanLab
        swanlab.save(best_model_path)
        
        # 完成实验
        swanlab.finish()
    
    print(f"训练完成！最佳F1分数: {best_val_f1}, 最佳模型已保存到: {best_model_path}")
    print(f"顶层最佳模型路径: {top_best_model_path}")


if __name__ == '__main__':
    # 定义模型训练的起始点
    best_val_f1 = 0.0
    main()