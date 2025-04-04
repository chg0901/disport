#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无序蛋白质区域预测竞赛提交脚本
用于比赛评测的预测脚本，符合比赛提交要求
"""

import os
import argparse
import torch
import csv
import pickle
import time
import logging
import numpy as np
from omegaconf import OmegaConf
from baseline import DisProtModel, restypes
from tqdm import tqdm
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("prediction")

def load_model(model_path, config_path):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型权重文件路径
        config_path: 配置文件路径
    
    返回:
        加载好的模型
    """
    logger.info(f"加载配置文件: {config_path}")
    config = OmegaConf.load(config_path)
    
    logger.info(f"初始化模型架构")
    model = DisProtModel(config.model)
    
    logger.info(f"加载模型权重: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, config

def preprocess_sequence(sequence):
    """
    预处理输入的蛋白质序列
    
    参数:
        sequence: 蛋白质序列（氨基酸单字母代码字符串）
    
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

def predict_sequence(model, sequence, device='cpu', batch_size=1000):
    """
    对单个序列进行预测，支持超长序列分批处理
    
    参数:
        model: 加载的模型
        sequence: 输入的蛋白质序列
        device: 计算设备
        batch_size: 分批处理的长度
    
    返回:
        预测结果字符串 (0和1组成的字符串)
    """
    # 检查序列长度
    seq_len = len(sequence)
    
    # 设置最大长度限制，与模型中的位置编码最大长度相匹配
    # 如果序列超过这个长度，需要分批处理
    max_position_len = 49000  # 略小于模型中的50000，留出安全边界
    
    # 如果序列不是很长，直接预测
    if seq_len <= max_position_len:
        # 预处理序列
        seq_tensor = preprocess_sequence(sequence).to(device)
        
        # 预测
        with torch.no_grad():
            pred = model(seq_tensor)
        
        # 获取每个位置的预测标签（0表示有序，1表示无序）
        pred_labels = torch.argmax(pred, dim=-1).squeeze().cpu().numpy()
        
        # 转换为比赛要求的字符串格式
        return ''.join([str(int(label)) for label in pred_labels])
    else:
        # 对于超长序列，分批处理
        logger.info(f"序列长度为 {seq_len}，超过 {max_position_len}，进行分批处理")
        
        # 设置更小的批处理大小，确保不会超过位置编码的最大长度
        effective_batch_size = min(batch_size, max_position_len)
        
        # 分割成多个子序列
        results = []
        for i in range(0, seq_len, effective_batch_size):
            end = min(i + effective_batch_size, seq_len)
            sub_seq = sequence[i:end]
            logger.info(f"处理子序列片段: 位置 {i} 到 {end-1}，长度: {len(sub_seq)}")
            
            seq_tensor = preprocess_sequence(sub_seq).to(device)
            
            with torch.no_grad():
                pred = model(seq_tensor)
            
            pred_labels = torch.argmax(pred, dim=-1).squeeze().cpu().numpy()
            results.append(pred_labels)
        
        # 合并结果
        all_results = np.concatenate(results)
        return ''.join([str(int(label)) for label in all_results])

def load_test_data(data_path):
    """
    加载测试数据
    
    参数:
        data_path: 测试数据文件路径
    
    返回:
        测试数据列表，每项包含蛋白质ID和序列
    """
    logger.info(f"尝试加载测试数据: {data_path}")
    test_data = []
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"测试数据文件不存在: {data_path}")
    
    # 尝试按照pickle格式加载
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            logger.info(f"成功以pickle格式加载数据，包含 {len(data)} 个样本")
            for i, item in enumerate(data):
                test_data.append({
                    'proteinID': f'test{i+1}' if 'id' not in item else item['id'],
                    'sequence': item['sequence']
                })
            return test_data
    except Exception as e:
        logger.info(f"以pickle格式加载失败: {str(e)}")
    
    # 尝试按照CSV格式加载
    try:
        with open(data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过标题行
            logger.info(f"CSV文件标题行: {header}")
            
            if len(header) >= 2:  # 至少包含蛋白质ID和序列
                count = 0
                for row in reader:
                    if len(row) >= 2:
                        test_data.append({
                            'proteinID': row[0],
                            'sequence': row[1]
                        })
                        count += 1
                logger.info(f"成功以CSV格式加载数据，包含 {count} 个样本")
                return test_data
    except Exception as e:
        logger.info(f"以CSV格式加载失败: {str(e)}")
    
    # 尝试按照文本文件格式加载（假设每行一个序列，格式为ID,sequence）
    try:
        with open(data_path, 'r') as f:
            lines = f.readlines()
            count = 0
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    test_data.append({
                        'proteinID': parts[0],
                        'sequence': parts[1]
                    })
                else:
                    # 如果没有ID，则使用行号作为ID
                    test_data.append({
                        'proteinID': f'test{i+1}',
                        'sequence': line.strip()
                    })
                count += 1
            logger.info(f"成功以文本格式加载数据，包含 {count} 个样本")
            return test_data
    except Exception as e:
        logger.info(f"以文本格式加载失败: {str(e)}")
    
    raise ValueError(f"无法识别测试数据文件格式: {data_path}")

def main():
    parser = argparse.ArgumentParser(description='蛋白质无序区域预测比赛提交')
    parser.add_argument('--model_path', default='./outputs/best_model.pth', help='模型权重文件路径')
    parser.add_argument('--config_path', default='./config.yaml', help='配置文件路径')
    parser.add_argument('--input_dir', default='/saisdata', help='输入数据目录')
    parser.add_argument('--output_dir', default='/saisresult', help='输出结果目录')
    parser.add_argument('--input_file', default='test.pkl', help='测试文件名')
    parser.add_argument('--output_file', default='submit.csv', help='输出文件名，默认为submit.csv')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='计算设备')
    parser.add_argument('--batch_size', type=int, default=1000, help='分批处理的长度')
    
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info(f"开始预测过程")
    
    try:
        # 检查设备可用性
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("警告: CUDA不可用，使用CPU替代")
            device = 'cpu'
        
        logger.info(f"使用设备: {device}")
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        input_path = os.path.join(args.input_dir, args.input_file)
        output_path = os.path.join(args.output_dir, args.output_file)
        
        logger.info(f"加载模型: {args.model_path}")
        model, config = load_model(args.model_path, args.config_path)
        model = model.to(device)
        
        logger.info(f"加载测试数据: {input_path}")
        test_data = load_test_data(input_path)
        
        logger.info(f"对 {len(test_data)} 个蛋白质序列进行预测...")
        
        # 创建输出CSV文件
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写入标题行
            writer.writerow(['proteinID', 'sequence', 'IDRs'])
            
            # 对每个蛋白质序列进行预测
            for idx, item in enumerate(tqdm(test_data, desc="预测进度")):
                protein_id = item['proteinID']
                sequence = item['sequence']
                
                try:
                    # 输出每个蛋白质序列的基本信息
                    logger.info(f"处理蛋白质 {idx+1}/{len(test_data)}: {protein_id}, 序列长度: {len(sequence)}")
                    
                    # 预测
                    prediction = predict_sequence(model, sequence, device, args.batch_size)
                    
                    # 写入预测结果
                    writer.writerow([protein_id, sequence, prediction])
                    
                    # 确保立即写入文件（避免程序崩溃时丢失数据）
                    csvfile.flush()
                except Exception as e:
                    logger.error(f"处理蛋白质 {protein_id} 时出错: {str(e)}")
                    logger.error(traceback.format_exc())
                    # 防止单个序列错误导致整个预测失败
                    writer.writerow([protein_id, sequence, '0' * len(sequence)])
                    csvfile.flush()
        
        elapsed_time = time.time() - start_time
        logger.info(f"预测完成，结果已保存到: {output_path}")
        logger.info(f"预测用时: {elapsed_time:.2f} 秒")
    
    except Exception as e:
        logger.error(f"预测过程出错: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 