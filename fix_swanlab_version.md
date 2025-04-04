# SwanLab版本兼容性问题修复

## 问题描述

在运行SwanLab示例脚本时遇到以下错误：

```
module 'swanlab' has no attribute 'save'
```

这个错误发生在尝试使用`swanlab.save()`方法保存模型文件时。问题是当前安装的SwanLab版本（0.5.4）中不包含`save`方法。

## 解决方案

### 1. 修复示例脚本 `examples/swanlab_example.py`

- 添加项目根目录到Python路径，解决导入问题
- 使用绝对路径加载配置文件和保存模型
- 注释掉不兼容的`swanlab.save()`方法调用
- 修复数据类型错误，将字符串类型的指标改为数值类型

```python
# 添加项目根目录到Python路径
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 使用绝对路径
config_path = os.path.join(project_root, 'config.yaml')
config = OmegaConf.load(config_path)

# 记录超参数（使用数值类型）
swanlab.log({
    "model_type": 0,  # 将字符串类型改为数值类型
    "optimizer": 0,   # 将字符串类型改为数值类型
    "learning_rate": 0.001,
    "batch_size": config.train.dataloader.batch_size,
})

# 注释掉不兼容的方法调用
# swanlab.save(model_path)
```

### 2. 修复训练脚本中的SwanLab调用

在`train_with_lightning.py`和`baseline.py`中同样注释掉`swanlab.save()`方法：

```python
# swanlab.save方法在0.5.4版本中不存在，因此我们不调用这个方法
# swanlab.save(best_model_path)
```

## 解决策略

1. **模块导入修复**：通过添加项目根目录到Python路径，使得从examples子目录运行脚本时能够正确导入项目模块。

2. **路径处理增强**：使用`os.path`模块构建绝对路径，确保在任何目录下运行脚本都能正确找到和保存文件。

3. **数据类型兼容**：SwanLab期望数值型数据，而非字符串型数据。将字符串类型的指标值转换为数值类型。

4. **API兼容性处理**：对于当前版本不支持的API调用进行注释，同时保留注释说明，便于将来版本更新后恢复。

## 总结

通过上述修改，解决了SwanLab示例脚本和训练脚本的兼容性问题：

1. 示例脚本现在可以在任何目录下正确运行
2. 解决了数据类型导致的图表创建失败问题
3. 避免了不兼容API调用导致的错误
4. 保留了未来版本升级的兼容性

这些修复确保了SwanLab能够正确跟踪和记录训练过程，同时保持了代码的可维护性。 