# DisPort - 无序蛋白质预测工具

DisPort是一个用于预测蛋白质序列中无序区域的工具，基于Transformer架构实现。

## 项目结构

```
disport/
├── config.yaml                # 配置文件
├── src/                       # 源代码
│   ├── data/                  # 数据处理模块
│   │   └── dataset.py         # 数据集类和数据处理函数
│   ├── models/                # 模型定义
│   │   └── transformer.py     # Transformer模型实现
│   └── utils/                 # 工具函数
│       ├── config.py          # 配置加载和解析
│       └── metrics.py         # 评估指标计算
├── scripts/                   # 脚本
│   ├── train.py               # 训练脚本
│   └── predict.py             # 预测脚本
├── run_test.sh                # 测试脚本
├── checkpoints/               # 模型检查点保存目录
└── predictions/               # 预测结果保存目录
```

## 技术特点

- 使用Transformer编码器实现蛋白质序列二分类（有序/无序）
- 支持可变长度的氨基酸序列输入
- 高效的评估指标计算，支持填充值排除
- 模块化设计，易于扩展和修改
- 完整的训练和预测流程
- 详细的可视化结果

## 性能指标

在模拟数据集上，模型预测无序区域的性能指标如下：
- F1分数：约0.43
- 准确率：约0.43
- 精确率：约0.43
- 召回率：约0.43

## 快速开始

### 准备环境
```bash
# 创建conda环境
conda create -n disprot python=3.8
conda activate disprot

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy scikit-learn matplotlib pyyaml tqdm
```

### 运行测试
```bash
# 一键测试训练和预测
./run_test.sh
```

### 模型训练
```bash
# 使用自定义配置文件训练
python scripts/train.py
```

### 预测
```bash
# 使用训练好的模型进行预测
python scripts/predict.py
```

## 修复说明

以下是对项目代码的主要修复：

1. **数据处理优化**：
   - 增强了`DisProtDataset`类处理可变长度序列的能力
   - 添加了自定义`collate_fn`函数以正确批处理不同长度的序列
   - 改进了数据集加载函数以支持多种数据格式

2. **指标计算改进**：
   - 优化了`calculate_metrics`函数以排除填充值
   - 添加了更详细的类别评估指标（每个类别的精确率、召回率和F1值）
   - 增加了零除处理以提高指标计算的稳定性

3. **训练流程优化**：
   - 改进了`train_epoch`和`validate`函数以支持批量指标计算
   - 添加了进度条显示训练和验证进度
   - 实现了更可靠的早停机制

4. **预测功能增强**：
   - 添加了可视化预测结果的功能
   - 改进了测试评估函数以提供更详细的结果
   - 增加了结果保存功能

5. **配置系统升级**：
   - 添加了`PredictConfig`类以支持预测配置
   - 改进了`Config`类以更好地组织配置
   - 增强了配置加载功能以支持更多参数

## 注意事项

- 当前代码在PyTorch 1.x版本下进行了测试
- GPU加速需要CUDA支持
- 模型参数可通过配置文件进行调整 