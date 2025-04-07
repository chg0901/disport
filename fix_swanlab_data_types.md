# 修复SwanLab数据类型错误

## 问题描述

在使用SwanLab记录训练过程中，出现了以下数据类型错误：

```
swanlab: Data type error, key: best_model_filename, data type: str, expected: float
swanlab: Chart 'best_model_filename' creation failed. Reason: The expected value type for the chart 'best_model_filename' is one of int,float or BaseType, but the input type is str.
swanlab: Data type error, key: output_directory, data type: str, expected: float
swanlab: Chart 'output_directory' creation failed. Reason: The expected value type for the chart 'output_directory' is one of int,float or BaseType, but the input type is str.
```

## 原因分析

SwanLab的日志记录系统对数据类型有严格要求，特别是在创建可视化图表时。SwanLab期望记录的指标值是数值类型（int、float或bool），但代码中尝试记录字符串类型的值，如模型文件名、输出目录路径和时间戳等。

问题出现在以下几个文件中：
- `train_with_lightning.py`
- `baseline.py`
- `examples/swanlab_example.py`

## 解决方案

### 1. 创建安全的SwanLab工具类

创建了一个专用的SwanLab工具类 `src/utils/swanlab_utils.py`，提供安全的日志记录功能：

```python
def safe_log(metrics: Dict[str, Any], verbose: bool = True) -> bool:
    """
    安全记录指标到SwanLab，过滤掉不适合的数据类型
    """
    # 过滤指标，只保留数值类型
    filtered_metrics = {}
    skipped_metrics = []
    
    for key, value in metrics.items():
        # 检查值是否为可接受的数值类型
        if isinstance(value, (int, float, bool)):
            filtered_metrics[key] = value
        else:
            skipped_metrics.append((key, type(value).__name__))
    
    # 如果有跳过的指标且verbose为True，则打印详细信息
    if skipped_metrics and verbose:
        print("以下指标被跳过，因为它们不是数值类型:")
        for key, type_name in skipped_metrics:
            print(f"  - {key} (类型: {type_name})")
    
    # 记录过滤后的指标
    if filtered_metrics:
        swanlab.log(filtered_metrics)
        return True
    else:
        return False
```

### 2. 修改现有脚本使用安全的日志记录功能

#### 修改 `train_with_lightning.py`

```python
# 导入安全工具类
from src.utils.swanlab_utils import safe_log, finish_experiment

# 使用安全日志记录函数
metrics_to_log = {
    "best_val_f1": best_f1_score,
    "total_epochs": trainer.current_epoch + 1,
    "best_model_filename": best_model_filename,  # 会被过滤掉
    "run_timestamp": timestamp,                  # 会被过滤掉
    "output_directory": output_dir               # 会被过滤掉
}
safe_log(metrics_to_log, verbose=True)

# 使用安全的完成函数
finish_experiment()
```

#### 修改 `baseline.py`

```python
# 导入安全工具类
from src.utils.swanlab_utils import safe_log, finish_experiment

# 使用安全日志记录函数
metrics_to_log = {
    "final_val_f1": val_f1,
    "best_val_f1": best_val_f1,
    "total_epochs": config.train.epochs,
    "run_timestamp": timestamp,             # 会被过滤掉
    "output_directory": output_dir          # 会被过滤掉
}
safe_log(metrics_to_log, verbose=True)

# 使用安全的完成函数
finish_experiment()
```

#### 修改 `examples/swanlab_example.py`

```python
# 导入安全工具类
from src.utils.swanlab_utils import safe_log, finish_experiment

# 使用安全日志记录函数
metrics_to_log = {
    "model_type": 0,  # 数值类型
    "optimizer": 0,   # 数值类型
    "learning_rate": 0.001,
    "batch_size": config.train.dataloader.batch_size,
    "model_name": "DisProtModel",  # 字符串类型，会被过滤掉
}
safe_log(metrics_to_log)
```

## 结果

修改后，SwanLab正常记录所有数值类型的指标，自动过滤掉不适合的字符串类型，同时提供清晰的日志，说明哪些字段被过滤以及原因。测试结果显示不再有数据类型错误，系统正常运行。

## 最佳实践

在使用SwanLab进行实验跟踪时，应遵循以下最佳实践：

1. 只记录数值类型的指标（int、float、bool）
2. 对于字符串类型的信息，使用常规日志记录而非SwanLab的图表功能
3. 使用安全的日志记录工具类来避免类型错误
4. 分离指标记录和文本记录，确保可视化图表的正确生成

这种方法确保了与SwanLab的平滑集成，提供了有价值的训练指标可视化，同时避免了数据类型错误。 