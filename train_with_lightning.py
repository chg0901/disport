import argparse
import os
import pickle
import datetime

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

# 设置PyTorch的float32矩阵乘法精度为'medium'以优化性能,这将通过牺牲一点精度来显著提高训练速度。
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
        try:
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
        except Exception as e:
            # 记录错误信息，但不中断测试过程
            print(f"测试样本 {batch_idx} 处理出错: {str(e)}")
            # 返回一个dummy的结果，使测试过程可以继续
            # 仅当绝大多数样本正常处理时，这个策略才有效
            return {'test_loss': torch.tensor(0.0, device=self.device), 'test_f1': torch.tensor(0.0, device=self.device)}
    
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
    parser.add_argument('--output_dir', default='./outputs', help='输出目录')
    parser.add_argument('--use_swanlab', action='store_true', help='是否使用SwanLab进行实验跟踪')
    parser.add_argument('--swanlab_project', default='disprot-lightning', help='SwanLab项目名称')
    parser.add_argument('--no_timestamp', action='store_true', help='不添加时间戳到输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config_path)
    
    # 添加时间戳到输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if not args.no_timestamp:
        output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
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
        batch_size=1,  # 使用较小的批大小，每次只处理一个样本，避免序列长度过长问题
        shuffle=False,
        num_workers=config.train.dataloader.num_workers
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
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='disprot-{epoch:02d}-f1_{val_f1:.4f}',
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
        save_dir=output_dir,
        name='disprot_logs'
    )
    loggers.append(tb_logger)
    
    # SwanLab日志记录器（如果需要）
    if args.use_swanlab:
        # 将配置转换为字典以便SwanLab记录
        config_dict = OmegaConf.to_container(config, resolve=True)
        
        # 添加运行时间戳和输出目录到配置
        config_dict['run_timestamp'] = timestamp
        config_dict['output_directory'] = output_dir
        
        # 创建SwanLab日志记录器
        swan_logger = SwanLabLogger(
            project=args.swanlab_project,
            config=config_dict,
            description=f"无序蛋白质区域预测模型训练（Lightning版本）- {timestamp}",
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
        deterministic=False,  # 关闭确定性模式，解决nll_loss2d_forward_out_cuda_template错误,交叉熵损失函数在CUDA上没有确定性实现。
        gradient_clip_val=1.0,  # 梯度裁剪，防止梯度爆炸
    )
    
    # 训练模型
    trainer.fit(model, train_dataloader, valid_dataloader)
    
    # 测试模型
    trainer.test(model, test_dataloader)
    
    # 打印最佳模型路径
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best F1 score: {checkpoint_callback.best_model_score:.4f}")
    
    # 复制最佳模型到outputs目录的顶层，以便于预测脚本使用
    best_model_path = checkpoint_callback.best_model_path
    best_model_filename = os.path.basename(best_model_path)
    best_f1_score = checkpoint_callback.best_model_score.item()
    
    # 创建包含F1分数和时间戳的文件名
    target_path = os.path.join(args.output_dir, f"best_model_f1_{best_f1_score:.4f}_{timestamp}.pth")
    
    # 复制最佳模型到顶层输出目录
    import shutil
    shutil.copy2(best_model_path, target_path)
    print(f"最佳模型已复制到: {target_path}")
    
    # 同时创建一个固定名称的副本，方便预测脚本使用
    standard_path = os.path.join(args.output_dir, "best_model.pth")
    shutil.copy2(best_model_path, standard_path)
    print(f"标准命名最佳模型已复制到: {standard_path}")
    
    # 保存最佳模型到SwanLab（如果启用）
    if args.use_swanlab:
        # swanlab.save方法在0.5.4版本中不存在，因此我们不调用这个方法
        # swanlab.save(best_model_path)
        
        # 记录最终结果
        swanlab.log({
            "best_model_filename": best_model_filename,
            "best_val_f1": best_f1_score,
            "total_epochs": trainer.current_epoch + 1,
            "run_timestamp": timestamp,
            "output_directory": output_dir
        })

        # 完成实验
        swanlab.finish()


if __name__ == '__main__':
    main() 