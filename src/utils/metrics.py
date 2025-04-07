"""
评估指标模块，用于计算无序蛋白质预测的性能指标
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def metric_fn(pred, gt):
    """
    计算F1分数
    
    参数:
        pred: 模型预测结果，shape为[batch_size, seq_len, num_classes]
        gt: 标签，shape为[batch_size, seq_len]
        
    返回:
        F1分数
    """
    # 将预测结果转换为类别索引
    pred_labels = torch.argmax(pred, dim=2).cpu()
    gt_labels = gt.cpu()
    
    # 创建掩码，排除填充值（通常为0）
    mask = (gt_labels != 0)
    
    # 仅使用非填充值计算F1
    pred_flat = pred_labels[mask].contiguous().view(-1)
    gt_flat = gt_labels[mask].contiguous().view(-1)
    
    # 计算微平均F1分数
    if len(gt_flat) > 0:
        f1 = f1_score(gt_flat, pred_flat, average='micro')
    else:
        f1 = 0.0
    
    return f1

def calculate_metrics(pred_labels, gt_labels):
    """
    计算多个评估指标
    
    参数:
        pred_labels: 模型预测的类别，shape为[batch_size, seq_len]
        gt_labels: 标签，shape为[batch_size, seq_len]
        
    返回:
        包含各项指标的字典
    """
    # 确保输入在CPU上
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu()
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.cpu()
    
    # 创建掩码，排除填充值（通常为0）
    mask = (gt_labels != 0)
    
    # 如果没有非填充值，返回默认指标
    if not torch.any(mask):
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'class_0_precision': 0.0,
            'class_0_recall': 0.0,
            'class_0_f1': 0.0,
            'class_1_precision': 0.0,
            'class_1_recall': 0.0,
            'class_1_f1': 0.0
        }
    
    # 仅使用非填充值计算指标
    pred_flat = pred_labels[mask].numpy()
    gt_flat = gt_labels[mask].numpy()
    
    # 计算各项指标
    metrics = {
        'accuracy': accuracy_score(gt_flat, pred_flat),
        'precision': precision_score(gt_flat, pred_flat, average='micro', zero_division=0),
        'recall': recall_score(gt_flat, pred_flat, average='micro', zero_division=0),
        'f1': f1_score(gt_flat, pred_flat, average='micro', zero_division=0)
    }
    
    # 计算每个类别的指标
    # 注意：如果某个类别没有样本，sklearn会发出警告，使用zero_division=0可以避免这个问题
    class_precision = precision_score(gt_flat, pred_flat, average=None, zero_division=0)
    class_recall = recall_score(gt_flat, pred_flat, average=None, zero_division=0)
    class_f1 = f1_score(gt_flat, pred_flat, average=None, zero_division=0)
    
    # 添加每个类别的指标
    for i, (p, r, f) in enumerate(zip(class_precision, class_recall, class_f1)):
        metrics[f'class_{i}_precision'] = p
        metrics[f'class_{i}_recall'] = r
        metrics[f'class_{i}_f1'] = f
    
    return metrics

def calculate_precision_recall_curve(predictions, targets, thresholds=None):
    """
    计算不同阈值下的精确率和召回率
    
    参数:
        predictions: 模型预测的概率值
        targets: 真实标签
        thresholds: 阈值列表，如果为None则使用默认范围
    
    返回:
        precision_list: 精确率列表
        recall_list: 召回率列表
        threshold_list: 阈值列表
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.05, 0.05)
    
    precision_list = []
    recall_list = []
    
    pred_flat = predictions.flatten().cpu().numpy()
    target_flat = targets.flatten().cpu().numpy()
    
    for threshold in thresholds:
        pred_binary = (pred_flat >= threshold).astype(np.int32)
        precision = precision_score(target_flat, pred_binary, zero_division=0)
        recall = recall_score(target_flat, pred_binary, zero_division=0)
        
        precision_list.append(precision)
        recall_list.append(recall)
    
    return precision_list, recall_list, thresholds


def calculate_auc(precision_list, recall_list):
    """
    计算PR曲线下的面积（AUC）
    
    参数:
        precision_list: 精确率列表
        recall_list: 召回率列表
    
    返回:
        auc: PR曲线下的面积
    """
    auc = 0.0
    
    # 按召回率排序
    sorted_indices = np.argsort(recall_list)
    sorted_recall = np.array(recall_list)[sorted_indices]
    sorted_precision = np.array(precision_list)[sorted_indices]
    
    # 计算PR-AUC
    for i in range(1, len(sorted_recall)):
        auc += (sorted_recall[i] - sorted_recall[i-1]) * (sorted_precision[i] + sorted_precision[i-1]) / 2
    
    return auc 