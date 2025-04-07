"""
SwanLab示例脚本
展示如何将SwanLab与无序蛋白质预测模型集成使用
"""

import os
import sys

# 添加项目根目录到Python路径，解决导入问题
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在目录的路径
current_dir = os.path.dirname(current_file_path)
# 获取项目根目录路径（当前目录的父目录）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到Python路径
sys.path.append(project_root)

import swanlab
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from baseline import DisProtModel, make_dataset
from omegaconf import OmegaConf
# 导入我们新创建的SwanLab工具类
from src.utils.swanlab_utils import safe_log, finish_experiment

# 加载配置
config_path = os.path.join(project_root, 'config.yaml')
config = OmegaConf.load(config_path)

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
metrics_to_log = {
    "model_type": 0,  # 数值类型
    "optimizer": 0,   # 数值类型
    "learning_rate": 0.001,
    "batch_size": config.train.dataloader.batch_size,
    "model_name": "DisProtModel",  # 字符串类型，会被过滤掉
}
safe_log(metrics_to_log)

# 模拟训练和评估过程
for epoch in range(5):
    # 模拟训练损失
    train_loss = 0.5 - 0.1 * epoch + 0.02 * np.random.randn()
    
    # 模拟验证指标
    val_loss = 0.6 - 0.08 * epoch + 0.03 * np.random.randn()
    val_f1 = 0.7 + 0.05 * epoch + 0.01 * np.random.randn()
    
    # 记录指标到SwanLab
    metrics_to_log = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_f1": val_f1,
    }
    safe_log(metrics_to_log)
    
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

# 保存示例模型权重
model_path = os.path.join(project_root, "model_demo.pth")
torch.save(model.state_dict(), model_path)
# 注意：0.5.4版本的SwanLab可能不支持save方法，因此注释掉这一行
# swanlab.save(model_path)

# 完成实验
finish_experiment()

print("SwanLab示例运行完成！请前往SwanLab查看训练可视化结果。") 


########################################################################################
################### 输出脚本  ########################################################### 
########################################################################################

# (disprot) cine@cine-WS-C621E-SAGE-Series:~/Documents/Github/disport$ python examples/swanlab_example.py
# swanlab: Tracking run with swanlab version 0.5.4                                                    
# swanlab: Run data will be saved locally in /home/cine/Documents/Github/disport/swanlog/run-20250405_030544-45ce3d3f
# swanlab: 👋 Hi chg0901, welcome to swanlab!
# swanlab: Syncing run ox-3 to the cloud
# swanlab: 🏠 View project at https://swanlab.cn/@chg0901/disprot-swanlab-demo
# swanlab: 🚀 View run at https://swanlab.cn/@chg0901/disprot-swanlab-demo/runs/07ekmimzfowu0dqyitp77
# Epoch 0: Train Loss: 0.5153, Val Loss: 0.6350, Val F1: 0.6978
# Epoch 1: Train Loss: 0.4020, Val Loss: 0.5028, Val F1: 0.7563
# Epoch 2: Train Loss: 0.2935, Val Loss: 0.4586, Val F1: 0.8090
# Epoch 3: Train Loss: 0.2007, Val Loss: 0.3372, Val F1: 0.8356
# Epoch 4: Train Loss: 0.1093, Val Loss: 0.3256, Val F1: 0.9043
# swanlab: 🏠 View project at https://swanlab.cn/@chg0901/disprot-swanlab-demo
# swanlab: 🚀 View run at https://swanlab.cn/@chg0901/disprot-swanlab-demo/runs/07ekmimzfowu0dqyitp77
# SwanLab示例运行完成！请前往SwanLab查看训练可视化结果。 