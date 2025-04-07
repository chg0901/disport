"""
训练器模块，包含无序蛋白质预测模型的训练逻辑
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
import numpy as np
import random
from tqdm import tqdm
import shutil

from src.utils.metrics import calculate_metrics, metric_fn
from src.data.dataset import make_dataset

# 尝试导入SwanLab
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False


class Trainer:
    """模型训练器类，封装训练和评估过程"""
    
    def __init__(self, config, model):
        """
        初始化训练器
        
        参数:
            config: 配置对象，包含训练参数
            model: 要训练的模型实例
        """
        self.config = config
        self.model = model
        self.setup_environment()
        self.setup_data()
        self.setup_training()
        self.setup_logging()
    
    def setup_environment(self):
        """配置训练环境，包括随机种子和设备"""
        # 设置随机种子以便结果可复现
        seed = self.config.train.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 设置计算设备
        self.device = torch.device("cuda" if self.config.train.use_gpu and torch.cuda.is_available() else "cpu")
        if self.config.train.use_gpu and torch.cuda.is_available():
            print(f"使用GPU: {self.config.train.gpu_ids}")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.config.train.gpu_ids))
            if len(self.config.train.gpu_ids) > 1:
                self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)
    
    def setup_data(self):
        """准备训练、验证和测试数据集"""
        # 加载数据集
        train_dataset, valid_dataset, test_dataset = make_dataset(
            self.config.data,
            train_rate=self.config.data.train_rate,
            valid_rate=self.config.data.valid_rate
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers
        )
        
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
    
    def setup_training(self):
        """配置训练参数，包括损失函数和优化器"""
        # 设置损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.train.optimizer.lr,
            weight_decay=self.config.train.optimizer.weight_decay
        )
        
        # 初始化训练状态
        self.start_epoch = 0
        self.best_metric = 0.0
        self.early_stop_count = 0
    
    def setup_logging(self):
        """配置日志和实验跟踪"""
        # 创建checkpoint目录
        os.makedirs(self.config.train.checkpoint_dir, exist_ok=True)
        
        # 配置SwanLab（如果可用且启用）
        self.swanlab_run = None
        if SWANLAB_AVAILABLE and self.config.swanlab and self.config.swanlab.enabled:
            self.swanlab_run = swanlab.init(
                experiment_name=self.config.swanlab.experiment_name,
                save_path=self.config.swanlab.save_path,
                config=self.config.__dict__
            )
    
    def train_epoch(self, epoch):
        """
        训练一个完整的epoch
        
        参数:
            epoch: 当前的epoch索引
            
        返回:
            训练损失和指标
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.train.epochs} [Train]")
        
        for sequences, labels in pbar:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            
            # 计算损失
            loss = self.criterion(outputs.permute(0, 2, 1), labels)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 记录结果
            total_loss += loss.item()
            
            # 收集预测和目标用于计算指标
            predictions = torch.argmax(outputs, dim=2).detach()
            all_predictions.append(predictions)
            all_targets.append(labels)
            
            # 更新进度条
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 计算平均损失和性能指标
        avg_loss = total_loss / len(self.train_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets)
        
        # 记录SwanLab指标（如果启用）
        if self.swanlab_run:
            self.swanlab_run.log({"train/loss": avg_loss})
            for metric_name, metric_value in metrics.items():
                self.swanlab_run.log({f"train/{metric_name}": metric_value})
        
        return avg_loss, metrics
    
    def validate(self, data_loader, phase="valid"):
        """
        在验证或测试集上评估模型
        
        参数:
            data_loader: 数据加载器
            phase: 阶段名称（"valid"或"test"）
            
        返回:
            验证损失和指标
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"[{phase.capitalize()}]")
            
            for sequences, labels in pbar:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(sequences)
                
                # 计算损失
                loss = self.criterion(outputs.permute(0, 2, 1), labels)
                
                # 记录结果
                total_loss += loss.item()
                
                # 收集预测和目标
                predictions = torch.argmax(outputs, dim=2)
                all_predictions.append(predictions)
                all_targets.append(labels)
                
                # 更新进度条
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 计算平均损失和性能指标
        avg_loss = total_loss / len(data_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets)
        
        # 记录SwanLab指标（如果启用）
        if self.swanlab_run:
            self.swanlab_run.log({f"{phase}/loss": avg_loss})
            for metric_name, metric_value in metrics.items():
                self.swanlab_run.log({f"{phase}/{metric_name}": metric_value})
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """
        保存模型检查点
        
        参数:
            epoch: 当前epoch
            metrics: 训练指标
            is_best: 是否是最佳模型
        """
        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备保存的状态
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # 保存常规检查点
        checkpoint_path = os.path.join(
            self.config.train.checkpoint_dir,
            f"checkpoint_epoch_{epoch+1}_{timestamp}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = os.path.join(
                self.config.train.checkpoint_dir,
                f"best_model_{timestamp}.pth"
            )
            torch.save(checkpoint, best_path)
            
            if self.swanlab_run:
                self.swanlab_run.log({
                    "best_epoch": epoch + 1,
                    "best_f1": metrics["f1"]
                })
    
    def train(self):
        """
        执行完整的训练过程
        
        返回:
            训练历史记录
        """
        print(f"开始训练，总共 {self.config.train.epochs} 个epochs")
        history = {
            'train_loss': [],
            'valid_loss': [],
            'train_metrics': [],
            'valid_metrics': []
        }
        
        for epoch in range(self.start_epoch, self.config.train.epochs):
            # 训练一个epoch
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # 验证
            valid_loss, valid_metrics = self.validate(self.valid_loader, "valid")
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_metrics'].append(train_metrics)
            history['valid_metrics'].append(valid_metrics)
            
            # 打印当前结果
            print(f"Epoch {epoch+1}/{self.config.train.epochs}")
            print(f"Train Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Valid Loss: {valid_loss:.4f}, F1: {valid_metrics['f1']:.4f}")
            
            # 检查是否是最佳模型
            is_best = valid_metrics['f1'] > self.best_metric
            if is_best:
                self.best_metric = valid_metrics['f1']
                self.early_stop_count = 0
                print(f"发现新的最佳模型！F1: {self.best_metric:.4f}")
            else:
                self.early_stop_count += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, valid_metrics, is_best)
            
            # 检查是否需要早停
            if self.early_stop_count >= self.config.train.early_stopping:
                print(f"早停条件满足 {self.early_stop_count} epochs没有改进。停止训练。")
                break
        
        # 测试最佳模型
        print("在测试集上评估最佳模型...")
        test_loss, test_metrics = self.validate(self.test_loader, "test")
        print(f"测试结果 - Loss: {test_loss:.4f}, F1: {test_metrics['f1']:.4f}")
        
        if self.swanlab_run:
            self.swanlab_run.log({
                "test/loss": test_loss,
                "test/f1": test_metrics['f1'],
                "test/accuracy": test_metrics['accuracy'],
                "test/precision": test_metrics['precision'],
                "test/recall": test_metrics['recall'],
                "test/mcc": test_metrics['mcc']
            })
        
        return history 