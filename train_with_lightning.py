import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import swanlab
from swanlab.integration.pytorch_lightning import SwanLabLogger

from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

from baseline import DisProtModel, DisProtDataset, make_dataset

# 设置PyTorch的float32矩阵乘法精度为'medium'以优化性能
torch.set_float32_matmul_precision('medium')

class DisProtLightningModel(pl.LightningModule):
    """
    使用PyTorch Lightning实现的无序蛋白质预测模型
    """
    def __init__(self, model_config, optimizer_config):
        """
        初始化Lightning模型
        
        参数:
            model_config: 模型配置
            optimizer_config: 优化器配置
        """
        super().__init__()
        # 保存超参数
        self.save_hyperparameters()
        
        # 创建模型
        self.model = DisProtModel(model_config)
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 保存优化器配置
        self.optimizer_config = optimizer_config

    def forward(self, x):
        """
        模型前向传播
        
        参数:
            x: 输入序列张量
        
        返回:
            模型输出
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        定义训练步骤
        
        参数:
            batch: 输入批次数据
            batch_idx: 批次索引
        
        返回:
            训练损失
        """
        sequence, label = batch
        pred = self(sequence)
        loss = self.loss_fn(pred.permute(0, 2, 1), label)
        
        # 记录训练损失
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        定义验证步骤
        
        参数:
            batch: 输入批次数据
            batch_idx: 批次索引
        
        返回:
            验证损失和预测结果
        """
        sequence, label = batch
        pred = self(sequence)
        loss = self.loss_fn(pred.permute(0, 2, 1), label)
        
        # 计算F1分数
        pred_labels = torch.argmax(pred, dim=-1).view(-1)
        gt_labels = label.view(-1)
        f1 = f1_score(y_true=gt_labels.cpu(), y_pred=pred_labels.cpu(), average='micro')
        
        # 记录验证指标
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_f1', f1, prog_bar=True, on_epoch=True, sync_dist=True)
        
        return {'val_loss': loss, 'val_f1': f1}
    
    def test_step(self, batch, batch_idx):
        """
        定义测试步骤
        
        参数:
            batch: 输入批次数据
            batch_idx: 批次索引
        
        返回:
            测试损失和预测结果
        """
        sequence, label = batch
        pred = self(sequence)
        loss = self.loss_fn(pred.permute(0, 2, 1), label)
        
        # 计算F1分数
        pred_labels = torch.argmax(pred, dim=-1).view(-1)
        gt_labels = label.view(-1)
        f1 = f1_score(y_true=gt_labels.cpu(), y_pred=pred_labels.cpu(), average='micro')
        
        # 记录测试指标
        self.log('test_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_f1', f1, prog_bar=True, on_epoch=True, sync_dist=True)
        
        return {'test_loss': loss, 'test_f1': f1}
    
    def configure_optimizers(self):
        """
        配置优化器
        
        返回:
            优化器
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay
        )
        
        # 添加学习率调度器（可选）
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def main():
    """
    主函数，解析参数并运行训练
    """
    parser = argparse.ArgumentParser('IDRs prediction with PyTorch Lightning')
    parser.add_argument('--config_path', default='./config.yaml')
    parser.add_argument('--max_epochs', type=int, default=None, help='最大训练轮数，不指定则使用配置文件中的值')
    parser.add_argument('--gpus', type=int, default=1, help='使用的GPU数量')
    parser.add_argument('--early_stopping', action='store_true', help='是否使用早停')
    parser.add_argument('--patience', type=int, default=5, help='早停的耐心值')
    parser.add_argument('--output_dir', default='./lightning_outputs', help='输出目录')
    parser.add_argument('--use_swanlab', action='store_true', help='是否使用SwanLab进行实验跟踪')
    parser.add_argument('--swanlab_project', default='disprot-lightning', help='SwanLab项目名称')
    
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config_path)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 最大训练轮数
    max_epochs = args.max_epochs if args.max_epochs is not None else config.train.epochs
    
    # 准备数据
    train_dataset, valid_dataset, test_dataset = make_dataset(config.data)
    
    # 数据加载器
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        **config.train.dataloader
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset, 
        batch_size=config.train.dataloader.batch_size, 
        shuffle=False
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=config.train.dataloader.batch_size, 
        shuffle=False
    )
    
    # 创建模型
    model = DisProtLightningModel(
        model_config=config.model,
        optimizer_config=config.train.optimizer
    )
    
    # 设置回调
    callbacks = []
    
    # 检查点保存
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='disprot-{epoch:02d}-{val_f1:.4f}',
        monitor='val_f1',
        mode='max',
        save_top_k=3,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # 早停（如果需要）
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_f1',
            min_delta=0.0001,
            patience=args.patience,
            verbose=True,
            mode='max'
        )
        callbacks.append(early_stop_callback)
    
    # 设置日志记录器
    loggers = []
    
    # 标准TensorBoard日志记录器
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='disprot_logs'
    )
    loggers.append(tb_logger)
    
    # SwanLab日志记录器（如果需要）
    if args.use_swanlab:
        # 将配置转换为字典以便SwanLab记录
        config_dict = OmegaConf.to_container(config, resolve=True)
        
        # 创建SwanLab日志记录器
        swan_logger = SwanLabLogger(
            project=args.swanlab_project,
            config=config_dict,
            description="无序蛋白质区域预测模型训练（Lightning版本）",
        )
        loggers.append(swan_logger)
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus if torch.cuda.is_available() else None,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=10,
        deterministic=False,  # 关闭确定性模式，解决nll_loss2d_forward_out_cuda_template错误
        gradient_clip_val=1.0,  # 梯度裁剪，防止梯度爆炸
    )
    
    # 训练模型
    trainer.fit(model, train_dataloader, valid_dataloader)
    
    # 测试模型
    trainer.test(model, test_dataloader)
    
    # 打印最佳模型路径
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best F1 score: {checkpoint_callback.best_model_score:.4f}")
    
    # 保存最佳模型到SwanLab（如果启用）
    if args.use_swanlab:
        swanlab.save(checkpoint_callback.best_model_path)
        best_model_filename = os.path.basename(checkpoint_callback.best_model_path)
        
        # 记录最终结果
        swanlab.log({
            "best_model_filename": best_model_filename,
            "best_val_f1": checkpoint_callback.best_model_score.item(),
            "total_epochs": trainer.current_epoch + 1
        })

        # 完成实验
        swanlab.finish()


if __name__ == '__main__':
    main() 