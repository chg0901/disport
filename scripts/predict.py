#!/usr/bin/env python
"""
预测脚本，用于使用训练好的模型进行无序蛋白质预测
"""

import os
import sys
import argparse
import torch
import glob
import numpy as np
import time
import pickle

# 将项目根目录添加到路径中，确保可以导入src模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.transformer import Transformer
from src.data.dataset import make_dataset, collate_fn
from src.utils.config import load_config
from src.utils.metrics import calculate_metrics
from torch.utils.data import DataLoader


# 定义20种标准氨基酸的单字母缩写
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="无序蛋白质预测")
    parser.add_argument("--config", type=str, default="../config/default_config.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, help="模型检查点文件路径")
    parser.add_argument("--input", type=str, help="输入FASTA文件路径")
    parser.add_argument("--output_dir", type=str, default="predictions", help="输出目录路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="预测阈值")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    parser.add_argument("--no_gpu", action="store_true", help="禁用GPU")
    return parser.parse_args()


def find_best_model(checkpoint_dir):
    """
    在检查点目录中查找最佳模型
    
    参数:
        checkpoint_dir: 检查点目录路径
        
    返回:
        最佳模型文件路径
    """
    # 首先检查是否存在标准命名的最佳模型
    standard_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(standard_path):
        return standard_path
    
    # 检查带有F1分数的最佳模型
    best_models = glob.glob(os.path.join(checkpoint_dir, "best_model_f1_*.pth"))
    if best_models:
        # 按文件修改时间排序，返回最新的
        return sorted(best_models, key=os.path.getmtime)[-1]
    
    # 如果没有找到最佳模型，则检查最新的检查点
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if checkpoints:
        return sorted(checkpoints, key=os.path.getmtime)[-1]
    
    raise FileNotFoundError(f"未找到任何模型文件，请检查 {checkpoint_dir} 目录")


def load_fasta(fasta_file):
    """
    加载FASTA格式的蛋白质序列
    
    参数:
        fasta_file: FASTA文件路径
        
    返回:
        包含序列ID和序列的字典列表
    """
    sequences = []
    current_id = None
    current_sequence = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                if current_id:
                    sequences.append({
                        'id': current_id,
                        'sequence': ''.join(current_sequence)
                    })
                current_id = line[1:].split()[0]  # 获取序列ID
                current_sequence = []
            else:
                current_sequence.append(line)
    
    # 添加最后一个序列
    if current_id:
        sequences.append({
            'id': current_id,
            'sequence': ''.join(current_sequence)
        })
    
    return sequences


def preprocess_sequence(sequence):
    """
    预处理蛋白质序列，转换为模型输入
    
    参数:
        sequence: 蛋白质序列字符串
        
    返回:
        处理后的序列张量
    """
    # 构建氨基酸映射字典
    residue_mapping = {'X': 20}
    residue_mapping.update(dict(zip(restypes, range(len(restypes)))))
    
    # 创建one-hot编码
    seq_tensor = torch.zeros(len(sequence), len(residue_mapping))
    for i, c in enumerate(sequence):
        if c not in restypes:
            c = 'X'
        seq_tensor[i][residue_mapping[c]] = 1
    
    return seq_tensor.unsqueeze(0)  # 添加批维度


def predict_sequence(model, sequence, device):
    """
    预测单个序列
    
    参数:
        model: 预测模型
        sequence: 输入序列
        device: 计算设备
        
    返回:
        预测结果
    """
    # 预处理序列
    seq_tensor = preprocess_sequence(sequence).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(seq_tensor)
        predictions = torch.softmax(outputs, dim=2)  # 转换为概率
        pred_classes = torch.argmax(predictions, dim=2).squeeze(0)
        pred_probs = predictions[:, :, 1].squeeze(0)  # 获取类别1（无序）的概率
    
    return {
        'classes': pred_classes.cpu().numpy(),
        'probabilities': pred_probs.cpu().numpy()
    }


def predict_batch(model, sequences, max_batch_size, device):
    """
    批量预测多个序列
    
    参数:
        model: 预测模型
        sequences: 序列列表
        max_batch_size: 最大批大小
        device: 计算设备
        
    返回:
        预测结果列表
    """
    from tqdm import tqdm
    
    results = []
    
    # 使用tqdm显示进度
    for i in tqdm(range(0, len(sequences), max_batch_size), desc="预测中"):
        batch_sequences = sequences[i:i+max_batch_size]
        
        for seq_data in batch_sequences:
            result = predict_sequence(model, seq_data['sequence'], device)
            results.append({
                'id': seq_data['id'],
                'sequence': seq_data['sequence'],
                'predictions': result
            })
    
    return results


def visualize_prediction(result, threshold=0.5):
    """
    可视化预测结果
    
    参数:
        result: 预测结果
        threshold: 二分类阈值
        
    返回:
        可视化字符串
    """
    sequence = result['sequence']
    predictions = result['predictions']
    classes = predictions['classes']
    probabilities = predictions['probabilities']
    
    # 二值化预测结果
    binary_pred = (probabilities >= threshold).astype(int)
    
    # 计算无序比例
    disorder_ratio = np.mean(binary_pred)
    disorder_count = np.sum(binary_pred)
    
    # 创建可视化字符串
    visualization = []
    visualization.append(f"序列ID: {result['id']}")
    visualization.append(f"序列长度: {len(sequence)}")
    
    # 显示序列
    if len(sequence) > 100:
        visualization.append(f"序列: {sequence[:50]}...{sequence[-50:]}")
    else:
        visualization.append(f"序列: {sequence}")
    
    # 显示结构预测
    prediction_str = ''.join(['D' if p == 1 else 'O' for p in classes])
    if len(prediction_str) > 100:
        visualization.append(f"预测 (O=有序, D=无序): {prediction_str[:50]}...{prediction_str[-50:]}")
    else:
        visualization.append(f"预测 (O=有序, D=无序): {prediction_str}")
    
    # 显示统计信息
    visualization.append(f"无序区域比例: {disorder_ratio:.2f}")
    visualization.append(f"无序残基数量: {disorder_count} / {len(sequence)}")
    
    return '\n'.join(visualization)


def save_prediction(result, output_dir, threshold=0.5):
    """
    保存预测结果到文件
    
    参数:
        result: 预测结果
        output_dir: 输出目录
        threshold: 二分类阈值
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sequence = result['sequence']
    predictions = result['predictions']
    classes = predictions['classes']
    probabilities = predictions['probabilities']
    
    # 创建输出文件路径
    output_file = os.path.join(output_dir, f"{result['id']}.txt")
    
    # 保存结果
    with open(output_file, 'w') as f:
        f.write(f">序列ID: {result['id']}\n")
        f.write(f">序列长度: {len(sequence)}\n\n")
        
        f.write(">序列:\n")
        f.write(f"{sequence}\n\n")
        
        f.write(">预测类别 (0=有序, 1=无序):\n")
        f.write(f"{' '.join(map(str, classes))}\n\n")
        
        f.write(">预测概率 (无序区域的概率):\n")
        f.write(f"{' '.join([f'{p:.4f}' for p in probabilities])}\n\n")
        
        # 二值化预测结果
        binary_pred = (probabilities >= threshold).astype(int)
        f.write(f">二值化预测 (阈值={threshold}):\n")
        f.write(f"{' '.join(map(str, binary_pred))}\n\n")
        
        # 统计信息
        disorder_ratio = np.mean(binary_pred)
        disorder_count = np.sum(binary_pred)
        f.write(">统计信息:\n")
        f.write(f"无序区域比例: {disorder_ratio:.4f}\n")
        f.write(f"无序残基数量: {disorder_count} / {len(sequence)}\n")
    
    # 创建可视化FASTA文件
    viz_file = os.path.join(output_dir, f"{result['id']}_visual.fasta")
    with open(viz_file, 'w') as f:
        f.write(f">{result['id']} | 原始序列\n")
        f.write(f"{sequence}\n")
        
        prediction_str = ''.join(['D' if p == 1 else 'O' for p in classes])
        f.write(f">{result['id']} | 预测结果 (O=有序, D=无序)\n")
        f.write(f"{prediction_str}\n")
    
    return output_file, viz_file


def test_model_with_dataloader(model, test_dataloader, criterion, config, device):
    """在测试数据集上评估模型"""
    # 确保模型处于评估模式
    model.eval()
    total_loss = 0
    all_pred_labels = []
    all_labels = []
    batch_metrics = []
    
    # 创建进度条
    from tqdm import tqdm
    progress_bar = tqdm(test_dataloader, desc="测试中")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            sequences, labels = batch
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # 前向传播
            predictions = model(sequences)
            
            # 计算损失
            loss = criterion(predictions.view(-1, predictions.size(-1)), labels.view(-1))
            total_loss += loss.item()
            
            # 获取预测的类别
            pred_labels = torch.argmax(predictions, dim=2)
            
            # 保存预测结果和标签
            all_pred_labels.append(pred_labels)
            all_labels.append(labels)
            
            # 计算批次的指标
            metrics = calculate_metrics(pred_labels, labels)
            batch_metrics.append(metrics)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'f1': metrics['f1']
            })
    
    # 计算平均指标
    avg_metrics = {
        'loss': total_loss / len(test_dataloader)
    }
    
    # 合并所有批次的指标
    for key in batch_metrics[0].keys():
        avg_metrics[key] = sum(batch[key] for batch in batch_metrics) / len(batch_metrics)
    
    # 返回结果
    return {
        'metrics': avg_metrics,
        'predictions': all_pred_labels,
        'labels': all_labels
    }


def visualize_predictions(test_results, config, num_samples=5):
    """
    可视化预测结果
    
    参数:
        test_results: 测试结果
        config: 配置对象
        num_samples: 要可视化的样本数量
    """
    print(f"\n可视化 {num_samples} 个预测结果示例:")
    
    # 获取预测和标签
    all_pred_labels = test_results['predictions']
    all_labels = test_results['labels']
    
    # 确保输出目录存在
    os.makedirs(config.predict.output_dir, exist_ok=True)
    
    # 创建可视化输出文件
    vis_file_path = os.path.join(config.predict.output_dir, "visualization.txt")
    
    with open(vis_file_path, 'w') as f:
        # 选择几个随机样本进行可视化
        num_batches = len(all_pred_labels)
        if num_batches == 0:
            print("没有可视化的预测结果")
            return
        
        max_samples = min(num_samples, num_batches)
        import random
        batch_indices = random.sample(range(num_batches), max_samples)
        
        for batch_idx in batch_indices:
            pred_batch = all_pred_labels[batch_idx]
            label_batch = all_labels[batch_idx]
            
            batch_size = pred_batch.size(0)
            max_batch_samples = min(1, batch_size)
            sample_indices = random.sample(range(batch_size), max_batch_samples)
            
            for sample_idx in sample_indices:
                pred = pred_batch[sample_idx]
                label = label_batch[sample_idx]
                
                # 排除填充部分
                mask = (label != 0)
                active_length = mask.sum().item()
                
                if active_length == 0:
                    continue
                
                pred = pred[:active_length]
                label = label[:active_length]
                
                # 创建可视化字符串
                pred_str = ''.join(['D' if p == 1 else 'O' for p in pred])
                label_str = ''.join(['D' if l == 1 else 'O' for l in label])
                
                # 计算准确度
                correct = (pred == label).sum().item()
                accuracy = correct / active_length
                
                # 写入文件
                f.write(f"\n样本 {batch_idx}_{sample_idx} (长度: {active_length}):\n")
                f.write(f"预测: {pred_str}\n")
                f.write(f"实际: {label_str}\n")
                f.write(f"准确度: {accuracy:.4f} ({correct}/{active_length})\n")
                f.write("="*50 + "\n")
                
                # 打印到控制台
                print(f"\n样本 {batch_idx}_{sample_idx} (长度: {active_length}):")
                print(f"预测: {pred_str}")
                print(f"实际: {label_str}")
                print(f"准确度: {accuracy:.4f} ({correct}/{active_length})")
                print("="*30)
    
    print(f"\n可视化结果已保存至: {vis_file_path}")


def main():
    """主函数"""
    # 加载配置文件
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    config = load_config(config_path)
    
    # 配置设备
    device = torch.device(f"cuda:{config.train.gpu}" if torch.cuda.is_available() and config.train.gpu >= 0 else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    train_dataset, val_dataset, test_dataset = make_dataset(config)
    print(f"测试数据集样本数: {len(test_dataset)}")
    
    # 创建测试数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.train.dataloader.batch_size,
        shuffle=False,
        num_workers=config.train.dataloader.num_workers,
        collate_fn=collate_fn  # 使用自定义的collate_fn
    )
    
    # 创建模型
    in_dim = config.model.i_dim
    out_dim = config.model.o_dim
    model = Transformer(
        d_model=config.model.d_model,
        nhead=config.model.n_head,
        num_encoder_layers=config.model.n_layer,
        num_decoder_layers=0,  # 不使用解码器
        dim_feedforward=4 * config.model.d_model,
        dropout=0.1,
        in_dim=in_dim,
        out_dim=out_dim
    ).to(device)
    
    # 加载最佳模型
    model_path = os.path.join(config.train.checkpoint_dir, "best_model.pth")
    if os.path.exists(model_path):
        # 直接加载状态字典
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"模型加载成功: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return
    
    # 设置损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 测试模型
    start_time = time.time()
    test_results = test_model_with_dataloader(model, test_dataloader, criterion, config, device)
    end_time = time.time()
    
    # 打印测试结果
    print("\n测试结果:")
    metrics = test_results['metrics']
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"测试耗时: {end_time - start_time:.2f}秒")
    
    # 确保输出目录存在
    os.makedirs(config.predict.output_dir, exist_ok=True)
    
    # 保存预测结果
    output_path = os.path.join(config.predict.output_dir, "predictions.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(test_results, f)
    print(f"预测结果已保存至: {output_path}")
    
    # 尝试可视化一些预测结果
    if hasattr(config.predict, 'visualize') and config.predict.visualize:
        visualize_predictions(test_results, config)


if __name__ == "__main__":
    main() 