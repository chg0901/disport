#!/usr/bin/env python
"""
训练脚本，用于训练无序蛋白质预测模型
"""

import os
import sys
import argparse
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 将项目根目录添加到路径中，确保可以导入src模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.transformer import Transformer
from src.data.dataset import make_dataset, collate_fn
from src.utils.config import load_config
from src.utils.metrics import metric_fn, calculate_metrics
from scripts.predict import test_model_with_dataloader

# 尝试导入SwanLab
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("警告: SwanLab未安装，将不会记录实验指标")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="无序蛋白质预测模型训练")
    parser.add_argument("--config", type=str, default="../config/default_config.yaml", help="配置文件路径")
    parser.add_argument("--no_gpu", action="store_true", help="禁用GPU")
    parser.add_argument("--gpu_ids", type=str, default=None, help="指定GPU ID，如 '0,1'")
    parser.add_argument("--batch_size", type=int, default=None, help="批处理大小")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--no_swanlab", action="store_true", help="禁用SwanLab日志记录")
    return parser.parse_args()


def setup_environment(config, args):
    """配置训练环境，包括随机种子和设备"""
    # 更新配置（如果在命令行中提供了参数）
    if args.batch_size is not None:
        config.train.dataloader.batch_size = args.batch_size
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.lr is not None:
        config.train.optimizer.lr = args.lr
    if args.seed is not None:
        config.train.seed = args.seed
    if args.no_gpu:
        config.train.use_gpu = False
    if args.gpu_ids is not None:
        config.train.gpu_ids = [int(id) for id in args.gpu_ids.split(",")]
    if args.no_swanlab:
        config.swanlab.enabled = False

    # 设置随机种子以便结果可复现
    seed = config.train.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置计算设备
    if config.train.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if hasattr(config.train, "gpu_ids") and config.train.gpu_ids:
            gpu_ids_str = ",".join(map(str, config.train.gpu_ids))
            print(f"使用GPU: {gpu_ids_str}")
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    return device


def setup_swanlab(config):
    """初始化SwanLab实验跟踪"""
    if SWANLAB_AVAILABLE and config.swanlab.enabled:
        # 确保SwanLab日志目录存在
        os.makedirs(config.swanlab.save_path, exist_ok=True)
        
        # 将配置转换为字典
        config_dict = {
            "model": {k: v for k, v in vars(config.model).items()},
            "data": {k: v for k, v in vars(config.data).items()},
            "train": {
                "epochs": config.train.epochs,
                "optimizer": {k: v for k, v in vars(config.train.optimizer).items()},
                "dataloader": {k: v for k, v in vars(config.train.dataloader).items()},
                "early_stopping": config.train.early_stopping,
            }
        }
        
        # 初始化SwanLab
        run = swanlab.init(
            experiment_name=config.swanlab.experiment_name,
            save_path=config.swanlab.save_path,
            config=config_dict
        )
        print(f"SwanLab实验已初始化: {config.swanlab.experiment_name}")
        return run
    return None


def save_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best=False):
    """保存模型检查点"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备保存的状态
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # 保存常规检查点
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # 如果是最佳模型，额外保存一份
    if is_best:
        best_model_path = os.path.join(output_dir, f"best_model_f1_{metrics['f1']:.4f}.pth")
        torch.save(checkpoint, best_model_path)
        
        # 创建一个标准命名的副本，方便预测脚本使用
        standard_best_model_path = os.path.join(output_dir, "best_model.pth")
        torch.save(model.state_dict(), standard_best_model_path)
        
        print(f"保存最佳模型，F1: {metrics['f1']:.4f}")
        return best_model_path
    
    return checkpoint_path


def train_epoch(model, train_dataloader, optimizer, criterion, epoch, config, device):
    model.train()
    total_loss = 0
    batch_metrics = []
    
    # 使用tqdm显示进度条
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        sequences, labels = batch
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # 前向传播
        predictions = model(sequences)
        
        # 计算损失
        loss = criterion(predictions.view(-1, predictions.size(-1)), labels.view(-1))
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计损失
        total_loss += loss.item()
        
        # 获取预测的类别
        pred_labels = torch.argmax(predictions, dim=2)
        
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
        'loss': total_loss / len(train_dataloader)
    }
    
    # 合并所有批次的指标
    for key in batch_metrics[0].keys():
        avg_metrics[key] = sum(batch[key] for batch in batch_metrics) / len(batch_metrics)
    
    return avg_metrics


def validate(model, val_dataloader, criterion, config, device):
    model.eval()
    total_loss = 0
    batch_metrics = []
    
    with torch.no_grad():
        # 使用tqdm显示进度条
        progress_bar = tqdm(val_dataloader, desc="Validation")
        
        for batch_idx, batch in enumerate(progress_bar):
            sequences, labels = batch
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # 前向传播
            predictions = model(sequences)
            
            # 计算损失
            loss = criterion(predictions.view(-1, predictions.size(-1)), labels.view(-1))
            
            # 统计损失
            total_loss += loss.item()
            
            # 获取预测的类别
            pred_labels = torch.argmax(predictions, dim=2)
            
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
        'loss': total_loss / len(val_dataloader)
    }
    
    # 合并所有批次的指标
    for key in batch_metrics[0].keys():
        avg_metrics[key] = sum(batch[key] for batch in batch_metrics) / len(batch_metrics)
    
    return avg_metrics


def train(model, train_dataloader, val_dataloader, config, device, swanlab_run=None):
    """训练模型"""
    # 设置损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.optimizer.lr,
        weight_decay=config.train.optimizer.weight_decay
    )
    
    # 创建检查点目录
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    
    # 初始化训练状态
    best_val_f1 = 0.0
    early_stop_count = 0
    
    print(f"开始训练，总共 {config.train.epochs} 个epoch")
    
    # 训练循环
    for epoch in range(config.train.epochs):
        # 训练一个epoch
        train_metrics = train_epoch(model, train_dataloader, optimizer, criterion, epoch, config, device)
        
        # 每个评估间隔进行验证
        if (epoch + 1) % config.train.eval_interval == 0:
            val_metrics = validate(model, val_dataloader, criterion, config, device)
            
            # 打印当前结果
            print(f"Epoch {epoch+1}/{config.train.epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # 记录SwanLab指标
            if swanlab_run:
                swanlab_run.log({
                    "train/loss": train_metrics['loss'],
                    "train/f1": train_metrics['f1'],
                    "val/loss": val_metrics['loss'],
                    "val/f1": val_metrics['f1'],
                    "epoch": epoch + 1
                })
            
            # 检查是否是最佳模型
            is_best = val_metrics['f1'] > best_val_f1
            if is_best:
                best_val_f1 = val_metrics['f1']
                early_stop_count = 0
                
                # 保存最佳模型
                best_model_path = save_checkpoint(
                    model, optimizer, epoch, val_metrics, 
                    config.train.checkpoint_dir, is_best=True
                )
                print(f"  发现新的最佳模型! F1: {best_val_f1:.4f}")
            else:
                early_stop_count += 1
                print(f"  未改进，早停计数: {early_stop_count}/{config.train.early_stop_patience}")
        
        # 每个保存间隔保存模型
        if (epoch + 1) % config.train.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics,
                config.train.checkpoint_dir, is_best=False
            )
        
        # 检查早停条件
        if early_stop_count >= config.train.early_stop_patience:
            print(f"早停条件满足（{config.train.early_stop_patience}个epoch未改进）。停止训练。")
            break
    
    print(f"训练完成！最佳验证F1: {best_val_f1:.4f}")
    return best_val_f1


def main():
    """主函数"""
    # 加载配置文件
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device(f"cuda:{config.train.gpu}" if torch.cuda.is_available() and config.train.gpu >= 0 else "cpu")
    print(f"使用设备: {device}")
    
    # 创建检查点目录
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    
    # 配置SwanLab
    swanlab_run = None
    if hasattr(config.train, 'use_swanlab') and config.train.use_swanlab:
        try:
            import swanlab
            swanlab_run = swanlab.init(
                experiment_name="disprotein-transformer",
                config=config.__dict__
            )
            # 记录配置到SwanLab
            for key, value in config.__dict__.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, (int, float, str, bool)):
                            swanlab_run.config[f"{key}.{k}"] = v
            print("已配置SwanLab，将记录训练过程")
        except ImportError:
            print("SwanLab未安装，不会记录训练过程")
        except Exception as e:
            print(f"SwanLab初始化失败: {e}")
    
    # 加载数据集
    print("加载数据集...")
    train_dataset, val_dataset, test_dataset = make_dataset(config)
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.dataloader.batch_size,
        shuffle=config.train.dataloader.shuffle,
        num_workers=config.train.dataloader.num_workers,
        drop_last=config.train.dataloader.drop_last,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.train.dataloader.batch_size,
        shuffle=False,
        num_workers=config.train.dataloader.num_workers,
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.train.dataloader.batch_size,
        shuffle=False,
        num_workers=config.train.dataloader.num_workers,
        collate_fn=collate_fn
    )
    
    # 创建模型
    print("创建模型...")
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
    
    # 多GPU训练设置（如果可用）
    if device.type == "cuda" and torch.cuda.device_count() > 1 and len(config.train.gpu_ids) > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    
    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {num_params:,}")
    
    # 开始训练
    print("\n开始训练过程...")
    best_val_f1 = train(model, train_dataloader, val_dataloader, config, device, swanlab_run)
    
    # 在测试集上评估最佳模型
    print("\n在测试集上评估最佳模型...")
    
    # 加载最佳模型
    best_model_path = os.path.join(config.train.checkpoint_dir, "best_model.pth")
    
    if os.path.exists(best_model_path):
        print(f"加载最佳模型: {best_model_path}")
        # 加载模型状态字典（不是完整检查点）
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        # 评估测试集
        criterion = torch.nn.CrossEntropyLoss()
        test_results = test_model_with_dataloader(model, test_dataloader, criterion, config, device)
        test_metrics = test_results['metrics']
        
        print(f"\n测试集结果:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        
        # 记录测试指标到SwanLab
        if swanlab_run:
            swanlab_run.log({
                "test/loss": test_metrics['loss'],
                "test/f1": test_metrics['f1'],
                "test/accuracy": test_metrics['accuracy'],
                "test/precision": test_metrics['precision'],
                "test/recall": test_metrics['recall']
            })
            # 保存最佳模型
            try:
                swanlab.save(best_model_path)
                print(f"最佳模型已保存到SwanLab")
            except Exception as e:
                print(f"SwanLab模型保存失败: {e}")
    else:
        print(f"\n最佳模型文件 {best_model_path} 不存在，跳过测试评估")
    
    print("\n训练完成!")


if __name__ == "__main__":
    main() 