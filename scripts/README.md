# DisPort 脚本使用指南

本目录包含用于训练和预测无序蛋白质区域的脚本。这些脚本提供了对DisPort模型的直接命令行接口，方便用户快速训练和部署模型。

## 训练脚本

`train.py` 脚本用于训练无序蛋白质预测模型。该脚本支持配置文件驱动，并提供多种命令行选项以自定义训练过程。

### 基本用法

```bash
python train.py --config ../config/default_config.yaml
```

### 命令行参数

- `--config`: 配置文件路径，默认为 `../config/default_config.yaml`
- `--no_gpu`: 禁用GPU，即使可用也使用CPU
- `--gpu_ids`: 指定要使用的GPU ID，如 '0,1'
- `--batch_size`: 批处理大小，覆盖配置文件中的设置
- `--epochs`: 训练轮数，覆盖配置文件中的设置
- `--lr`: 学习率，覆盖配置文件中的设置
- `--seed`: 随机种子，覆盖配置文件中的设置
- `--no_swanlab`: 禁用SwanLab实验跟踪

### 示例

```bash
# 使用自定义配置文件训练
python train.py --config ../config/custom_config.yaml

# 使用CPU训练
python train.py --no_gpu

# 使用特定GPU
python train.py --gpu_ids 0,1

# 自定义训练参数
python train.py --batch_size 16 --epochs 100 --lr 0.0001
```

## 预测脚本

`predict.py` 脚本用于使用训练好的模型预测蛋白质序列中的无序区域。该脚本支持FASTA格式的输入文件，并提供详细的预测结果输出。

### 基本用法

```bash
python predict.py --checkpoint ../checkpoints/best_model.pth --input ../data/example.fasta
```

### 命令行参数

- `--config`: 配置文件路径，默认为 `../config/default_config.yaml`
- `--checkpoint`: 模型检查点文件路径
- `--input`: 输入FASTA文件路径
- `--output_dir`: 输出目录路径，默认为 `predictions`
- `--threshold`: 预测阈值，默认为0.5
- `--batch_size`: 批处理大小，默认为8
- `--no_gpu`: 禁用GPU，即使可用也使用CPU

### 示例

```bash
# 使用特定检查点预测
python predict.py --checkpoint ../checkpoints/model_epoch_30.pth --input ../data/test_sequences.fasta

# 自定义输出目录和阈值
python predict.py --input ../data/test_sequences.fasta --output_dir my_predictions --threshold 0.7

# 使用CPU预测
python predict.py --input ../data/test_sequences.fasta --no_gpu
```

## 输出文件

预测脚本会为每个输入序列生成两种输出文件：

1. `序列ID.txt`: 包含详细的预测结果，包括无序概率和二分类结果
2. `序列ID_visual.fasta`: FASTA格式的可视化文件，显示原始序列和预测结果（'O'表示有序，'D'表示无序）

## 性能提示

- 对于大型数据集，适当增加批大小可以提高训练速度
- 使用GPU可以显著加快训练和预测速度
- 对于超长序列，脚本会自动处理，但可能会增加内存使用量
- 如果出现内存不足的问题，可以尝试降低批大小

## 故障排除

- 如果出现CUDA内存错误，尝试减小批大小或使用CPU模式
- 如果模型性能不佳，可以尝试调整超参数，如学习率、模型大小或训练轮数
- 如果预测结果不符合预期，可以尝试调整预测阈值 