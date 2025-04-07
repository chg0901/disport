#!/bin/bash
set -e

echo "===== 无序蛋白质预测项目测试 ====="

# 当前目录为项目根目录
echo "当前目录: $(pwd)"

# 确保配置文件存在
if [ ! -f "config.yaml" ]; then
  echo "错误: 配置文件 config.yaml 不存在"
  exit 1
fi

# 创建所需目录
mkdir -p checkpoints predictions

# 生成测试数据
echo -e "\n===== 生成模拟数据集 ====="
python -c "
import os
import pickle
import numpy as np

# 生成模拟数据
print('生成模拟蛋白质数据集...')
num_samples = 100
max_seq_length = 50
sequences = []
labels = []

# 生成随机序列和标签
for i in range(num_samples):
    seq_length = np.random.randint(10, max_seq_length + 1)
    # 使用蛋白质字母表 A-Z (不包含 B, J, O, U, X, Z)
    aa_list = list('ACDEFGHIKLMNPQRSTVWY')
    sequence = np.random.choice(aa_list, size=seq_length)
    # 为每个位置生成标签 (0 表示有序，1 表示无序)
    label = np.random.randint(0, 2, size=seq_length)
    
    sequences.append(''.join(sequence))
    labels.append(label)

# 划分数据集
train_size = int(0.7 * num_samples)
val_size = int(0.2 * num_samples)
test_size = num_samples - train_size - val_size

train_indices = np.arange(0, train_size)
val_indices = np.arange(train_size, train_size + val_size)
test_indices = np.arange(train_size + val_size, num_samples)

# 保存数据集
data = {
    'train': {
        'sequences': [sequences[i] for i in train_indices],
        'labels': [labels[i] for i in train_indices]
    },
    'val': {
        'sequences': [sequences[i] for i in val_indices],
        'labels': [labels[i] for i in val_indices]
    },
    'test': {
        'sequences': [sequences[i] for i in test_indices],
        'labels': [labels[i] for i in test_indices]
    }
}

output_path = 'mock_data.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(data, f)

print(f'生成的数据集包含:')
print(f'  - 训练集: {len(data[\"train\"][\"sequences\"])} 个样本')
print(f'  - 验证集: {len(data[\"val\"][\"sequences\"])} 个样本')
print(f'  - 测试集: {len(data[\"test\"][\"sequences\"])} 个样本')
print(f'数据集已保存到: {os.path.abspath(output_path)}')
"

# 更新配置文件中的数据路径
echo -e "\n===== 更新配置文件 ====="
sed -i 's|data_path:.*|data_path: "mock_data.pkl"|' config.yaml

# 显示配置文件内容
echo "配置文件内容:"
cat config.yaml

# 训练模型
echo -e "\n===== 开始训练模型 ====="
CUDA_VISIBLE_DEVICES=0 python scripts/train.py

# 预测
echo -e "\n===== 使用模型进行预测 ====="
CUDA_VISIBLE_DEVICES=0 python scripts/predict.py

echo -e "\n===== 测试完成 ====="
echo "检查点已保存在: $(pwd)/checkpoints"
echo "预测结果已保存在: $(pwd)/predictions" 