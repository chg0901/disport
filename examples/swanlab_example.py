"""
SwanLab示例脚本
展示如何将SwanLab与无序蛋白质预测模型集成使用
"""

import swanlab
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from baseline import DisProtModel, make_dataset
from omegaconf import OmegaConf

# 加载配置
config = OmegaConf.load('./config.yaml')

# 初始化SwanLab实验
swanlab.init(
    project="disprot-swanlab-demo",
    description="无序蛋白质区域预测模型示例",
    config=OmegaConf.to_container(config, resolve=True)
)

# 加载数据
train_dataset, valid_dataset, test_dataset = make_dataset(config.data)

# 创建一个简化版的训练循环
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DisProtModel(config.model).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 记录超参数和模型架构概要
swanlab.log({
    "model_type": "Transformer",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": config.train.dataloader.batch_size,
})

# 模拟训练和评估过程
for epoch in range(5):
    # 模拟训练损失
    train_loss = 0.5 - 0.1 * epoch + 0.02 * np.random.randn()
    
    # 模拟验证指标
    val_loss = 0.6 - 0.08 * epoch + 0.03 * np.random.randn()
    val_f1 = 0.7 + 0.05 * epoch + 0.01 * np.random.randn()
    
    # 记录指标到SwanLab
    swanlab.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_f1": val_f1,
    })
    
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

# 保存示例模型权重
torch.save(model.state_dict(), "model_demo.pth")
swanlab.save("model_demo.pth")

# 完成实验
swanlab.finish()

print("SwanLab示例运行完成！请前往SwanLab查看训练可视化结果。") 