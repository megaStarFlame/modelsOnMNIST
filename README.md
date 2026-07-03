# Reproduction and Training of Classic CNN Models on MNIST/FashionMNIST

This project reproduces classic convolutional neural networks (AlexNet, VGGNet, GoogLeNet, ResNet, MobileNet) based on PyTorch, and completes training and validation on the MNIST handwritten digit dataset (and FashionMNIST clothing dataset). It includes complete training pipelines, accuracy/loss recording, and model structure modifications.


## Project Structure

├── AlexNet.py # AlexNet model (modified version with BN layers)
├── VGGNet.py # VGGNet model (lightweight modified version)
├── GoogLeNet.py # GoogLeNet model (modified version with BN layers)
├── ResNet.py # ResNet residual network implementation
├── MobileNet.py # MobileNet depthwise separable convolution
├── common_methods.py # Common tools: data loading, training/evaluation, result saving
├── main.py # Main training script: batch train different models
└── README.md # Project description


## Core Features
1. Reproduce 5 classic CNN models; some models add BN layers/lightweight modifications to adapt to small datasets;
2. Automatically load MNIST/FashionMNIST datasets, support image resize (adapt to 224×224 input);
3. Configure optimizers (SGD/Adam) per model, automatically train and save loss/accuracy per epoch to CSV files;
4. Fix random seeds for reproducibility, support automatic GPU/CPU switching.

## Environment Dependencies
```bash
# Core dependencies
torch >= 1.10.0
torchvision >= 0.11.0
numpy >= 1.21.0
matplotlib >= 3.4.0
d2l >= 1.0.0  # Optional, for d2l.torch tools
```

Installation command:
```bash
pip install torch torchvision numpy matplotlib d2l
```

Quick Start
1. Configure Paths
Modify paths in common_methods.py (recommended to use environment variables to avoid hardcoding):
```python
# Method 1: Modify placeholders (temporary)
data_path = "./data"       # Dataset download/storage path
local_path = "./model_log" # Training log/CSV save path

# Method 2: Use environment variables (recommended)
import os
data_path = os.getenv("MNIST_DATA_PATH", "./data")
local_path = os.getenv("MODEL_SAVE_PATH", "./model_log")
```

2. Run Training
Execute the main script to train all models in batch:
```bash
python main.py
```

During training, the following will be output:
Training/test accuracy and loss for each model;
Generate a CSV file for each model in local_path (recording epoch, train_loss, train_acc, test_acc).
3. Custom Training
Modify the loop_train list in main.py to adjust:

```python
loop_train = [
    ('AlexNet',128,AlexNet.AlexNet_modify,torch.optim.SGD),  # (model_name, batch_size, model_class, optimizer)
    ('VGG',64,VGGNet.VGG_modify,torch.optim.SGD),
    # Add/remove/modify models, batch_size, optimizers as needed
]
```

Model Modification Notes

| Model	    |Modifications	|Optimizer|
| ----------- | ----------- | ----------- |
| AlexNet	|Add BN layers, adjust convolution padding for 224×224 input	|SGD|
| VGGNet	|Lightweight channel size (256→128→64), add BN layers	|SGD|
| GoogLeNet	|Add BN layers to all branches	|Adam|
| ResNet	|Basic residual block structure, adapt to MNIST output dimension |Adam|
| MobileNet	|Depthwise separable convolution, adapt to small dataset	|Adam|

Notes
1. The MNIST dataset is single-channel (1×28×28), and all models are adapted to input channels=1 (original models default to 3 channels, modified);
2. Automatically detect GPU during training, switch to CPU if no GPU is available;
3. Adjust batch_size according to GPU memory (e.g., 64 for VGG, 128 for others);
4. Training log CSV files are appended automatically, delete old files before re-training.

Extension Directions
1. Replace with FashionMNIST: Modify load_mnist to load_fashion_mnist in common_methods.py;
2. Add learning rate schedulers (e.g., StepLR);
3. Add model inference/visualization functions;
4. Compare effects of different optimizers/learning rates.