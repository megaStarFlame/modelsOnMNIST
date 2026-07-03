# 经典CNN模型在MNIST/FashionMNIST上的复现与训练

该项目基于PyTorch复现了AlexNet、VGGNet、GoogLeNet、ResNet、MobileNet等经典卷积神经网络，并在MNIST手写数字数据集（及FashionMNIST服饰数据集）上完成训练与验证，包含完整的训练流程、精度/损失记录、模型结构修改等功能。

## 项目结构

├── AlexNet.py # AlexNet 模型（含修改版，添加 BN 层）
├── VGGNet.py # VGGNet 模型（含轻量化修改版）
├── GoogLeNet.py # GoogLeNet 模型（含 BN 层修改版）
├── ResNet.py # ResNet 残差网络实现
├── MobileNet.py # MobileNet 深度可分离卷积实现
├── common_methods.py # 通用工具：数据加载、训练 / 评估、结果保存
├── main.py # 主训练脚本：批量训练不同模型
└── README.md # 项目说明


## 核心功能
1. 复现5种经典CNN模型，部分模型添加BN层/轻量化修改以适配小数据集；
2. 自动加载MNIST/FashionMNIST数据集，支持图片尺寸resize（适配224×224输入）；
3. 分模型配置优化器（SGD/Adam），自动训练并保存每轮epoch的损失/精度到CSV文件；
4. 固定随机种子保证实验可复现，支持GPU/CPU自动切换。

## 环境依赖
```python
# 核心依赖
torch >= 1.10.0
torchvision >= 0.11.0
numpy >= 1.21.0
matplotlib >= 3.4.0
d2l >= 1.0.0  # 可选，用于d2l.torch工具
```

安装命令：

```bash
pip install torch torchvision numpy matplotlib d2l
```

快速开始
1. 配置路径
修改common_methods.py中的路径：
```python
# 方式1：直接修改占位符（临时）
data_path = "./data"       # 数据集下载/存储路径
local_path = "./model_log" # 训练日志/CSV文件保存路径

# 方式2：通过环境变量（推荐）
import os
data_path = os.getenv("MNIST_DATA_PATH", "./data")
local_path = os.getenv("MODEL_SAVE_PATH", "./model_log")
```

2. 运行训练
直接执行主脚本，自动批量训练所有模型：
```bash
python main.py
```

训练过程中会输出：
每个模型的训练 / 测试精度、损失；
在local_path下生成每个模型的 CSV 文件（记录 epoch、train_loss、train_acc、test_acc）。
3. 自定义训练
修改main.py中的loop_train列表，可调整：
```python
loop_train = [
    ('AlexNet',128,AlexNet.AlexNet_modify,torch.optim.SGD),  # (模型名, batch_size, 模型类, 优化器)
    ('VGG',64,VGGNet.VGG_modify,torch.optim.SGD),
    # 按需增删/修改模型、batch_size、优化器
]
```

模型修改说明

| 模型	    |修改点	|优化器 |
| -------- | -------- | -------- |
| AlexNet	|添加 BN 层，调整卷积 padding 适配 224×224 输入	|SGD|
| VGGNet	|轻量化通道数（256→128→64），添加 BN 层	|SGD|
| GoogLeNet	|所有分支添加 BN 层	|Adam|
| ResNet	|基础残差块结构，适配 MNIST 输出维度	|Adam|
| MobileNet	|深度可分离卷积，适配小数据集	|Adam|

注意事项
1. MNIST 数据集为单通道（1×28×28），所有模型均适配输入通道数为 1（默认原模型为 3 通道，已修改）；
2. 训练时自动检测 GPU，无 GPU 则自动切换到 CPU；
3. 建议批量大小（batch_size）根据 GPU 显存调整（如 VGG 建议 64，其余 128）；
4. 训练日志 CSV 文件会自动追加，重复训练前可删除旧文件。

扩展方向
1. 替换为 FashionMNIST：修改common_methods.py中的load_mnist为load_fashion_mnist；
2. 增加学习率调度器（如 StepLR）；
3. 添加模型推理 / 可视化功能；
4. 对比不同优化器 / 学习率的效果。