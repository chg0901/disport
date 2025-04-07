"""
SwanLab工具模块，提供与SwanLab集成的辅助功能
"""

from typing import Dict, Any, Union, List, Optional
import warnings

# 尝试导入SwanLab
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    warnings.warn("SwanLab未安装，日志功能将被禁用。请安装SwanLab: pip install swanlab")


def safe_log(metrics: Dict[str, Any], verbose: bool = True) -> bool:
    """
    安全记录指标到SwanLab，过滤掉不适合的数据类型
    
    参数:
        metrics: 要记录的指标字典，键为指标名称，值为指标值
        verbose: 是否打印详细信息
        
    返回:
        bool: 是否成功记录
    """
    if not SWANLAB_AVAILABLE:
        if verbose:
            print("SwanLab未安装，无法记录指标")
        return False
    
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
            # 如果是字符串，安全打印其值
            if isinstance(metrics[key], str):
                safe_value = metrics[key][:50] + "..." if len(metrics[key]) > 50 else metrics[key]
                print(f"    值: '{safe_value}'")
    
    # 记录过滤后的指标
    if filtered_metrics:
        try:
            swanlab.log(filtered_metrics)
            if verbose:
                print(f"成功记录 {len(filtered_metrics)} 个指标到SwanLab")
            return True
        except Exception as e:
            if verbose:
                print(f"记录指标到SwanLab时出错: {str(e)}")
            return False
    else:
        if verbose:
            print("没有有效的指标可记录")
        return False


def finish_experiment(verbose: bool = True) -> bool:
    """
    安全地结束SwanLab实验
    
    参数:
        verbose: 是否打印详细信息
        
    返回:
        bool: 是否成功结束实验
    """
    if not SWANLAB_AVAILABLE:
        if verbose:
            print("SwanLab未安装，无需结束实验")
        return False
    
    try:
        swanlab.finish()
        if verbose:
            print("SwanLab实验已成功结束")
        return True
    except Exception as e:
        if verbose:
            print(f"结束SwanLab实验时出错: {str(e)}")
        return False 