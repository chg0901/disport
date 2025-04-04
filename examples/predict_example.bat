@echo off
SETLOCAL

REM 准备好示例序列和模型路径 (用户需要修改MODEL_PATH指向实际模型路径)
SET MODEL_PATH=.\outputs\checkpoints\best_model.ckpt
SET CONFIG_PATH=.\config.yaml
SET EXAMPLE_FASTA=.\examples\example.fasta
SET OUTPUT_FILE=.\examples\p53_prediction.txt

REM 检查模型文件是否存在
IF NOT EXIST "%MODEL_PATH%" (
    echo 错误: 模型文件不存在: %MODEL_PATH%
    echo 请先训练模型或修改MODEL_PATH指向您的模型文件
    goto END
)

REM 运行预测
echo 对p53蛋白质进行无序区域预测...
python predict.py ^
    --model_path "%MODEL_PATH%" ^
    --config_path "%CONFIG_PATH%" ^
    --sequence_file "%EXAMPLE_FASTA%" ^
    --output "%OUTPUT_FILE%"

echo 预测完成! 结果已保存到: %OUTPUT_FILE%
echo 您也可以查看控制台输出的可视化结果

REM 打印结果文件的前几行
echo.
echo 预览结果文件的内容:
type "%OUTPUT_FILE%" | findstr /N "^" | findstr /B "^[1-9]:" 

:END
ENDLOCAL 