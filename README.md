# 3D-ResNet18-with-Contrastive-Learning-Loss

This repository implements a **3D ResNet-18** model with a supervised **contrastive learning loss** function. 
The primary objective is to extract meaningful spatial and temporal features from video data, making the model 
suitable for tasks like video classification and action recognition.

## Features

- **3D ResNet-18 Architecture**: Utilizes 3D convolutions to handle spatial and temporal data effectively.
- **Supervised Contrastive Loss**: Encourages the model to cluster similar samples together in the embedding space 
  while separating dissimilar samples.
- **Custom Dataset Support**: Includes tools for loading and preprocessing video datasets.
- **Training Modes**: Supports training with contrastive loss or cross-entropy loss for different objectives.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sadafsf/3D-ResNet18-with-Contrastive-Learning-Loss.git
   cd 3D-ResNet18-with-Contrastive-Learning-Loss
   ```
---

## Repository Structure

```plaintext
.
├── resnet.py             # Defines the 3D ResNet architecture
├── main.py               # Main script for training and evaluation
├── train_lowlevel.py     # Training functions with loss calculation
├── spc.py                # Implementation of supervised contrastive loss
├── datasets.py           # Dataset loading and preprocessing
├── transforms.py         # Data augmentation techniques
├── requirements.txt      # Python dependencies
└── README.md             # Repository documentation (this file)
```

---

## Usage

### Training the Model
Run the `main.py` script to train the model. Customize the training parameters using command-line arguments.

```bash
python main.py --dataset_path /path/to/dataset --batch_size 16 --epochs 50 --learning_rate 0.001 --loss_mode contrastive
```

**Key Arguments:**
- `--dataset_path`: Path to the dataset directory.
- `--batch_size`: Batch size for training.
- `--epochs`: Number of training epochs.
- `--learning_rate`: Learning rate for the optimizer.
- `--loss_mode`: Choose between `contrastive` or `cross_entropy`.

### Dataset Preparation
Ensure your video dataset is structured in the following format:

```plaintext
dataset/
├── class1/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── class2/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── ...
```

The `datasets.py` file includes functions for loading videos, splitting them into frames, and applying necessary transformations.

### Custom Loss Functions
The repository includes an implementation of supervised contrastive loss in `spc.py`. This loss function can be customized for specific use cases.

---
