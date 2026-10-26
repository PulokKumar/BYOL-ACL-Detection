# BYOL-ACL Classifier

This repository provides a PyTorch implementation for **Anterior Cruciate Ligament (ACL) tear detection** using a ResNet-18 encoder with various initialization strategies, including **self-supervised pretraining with BYOL**, ImageNet pretraining, and random initialization.

## Dataset

We used the publicly available dataset:
**[ACL Tear MRI Dataset (Zenodo)](https://zenodo.org/records/14789903)**

The dataset includes MRI scans labeled into three classes:

- **Healthy**
- **Partial tear**
- **Complete tear**

The data is automatically split into training, validation, and test sets using ratios **70% / 15% / 15%**.

## Features

- Train a classifier with either a **linear** or **MLP head**.

- Fine-tuning with or without encoder freezing.

- Support for three initialization modes:

  - `byol`: Load weights from BYOL-pretrained encoder
  - `imagenet`: Load weights from ImageNet-pretrained encoder
  - `random`: Random initialization

- Automatic split into train/validation/test sets.

## Installation

```bash
git clone https://github.com/your-username/BYOL-ACL.git
cd BYOL-ACL
pip install -r requirements.txt
```

## Usage pre-train

### Arguments

### Arguments

- `--data_root_unl`: Path to unlabeled dataset root directory.
- `--epochs`: Number of training epochs (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)

### Example Command

```bash
python byol_pretrain.py --data_root_unl /path/to/unlabeled_data --seed 42 --epochs 100
```

## Output

The script will:

- Save pretrained encoder in `./pretrained/`.

## Usage fine-tune

### Arguments

- `--data_root`: Path to dataset root directory.
- `--head`: Classifier head type: `linear` or `mlp` (default: `mlp`)
- `--pretrained`: Encoder initialization: `byol`, `imagenet`, or `random` (default: `byol`)
- `--byol_encoder_ckpt`: Path to BYOL-pretrained encoder checkpoint (required if `--pretrained byol`)
- `--freeze`: Freeze encoder during fine-tuning (flag)
- `--epochs`: Number of training epochs (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)

### Example Command

```bash
python train.py \
  --data_root /path/to/acl_dataset \
  --head mlp \
  --pretrained byol \
  --byol_encoder_ckpt ./byol_encoder.pth \
  --freeze \
  --epochs 100
```

## Output

The script will:

- Train the model on the specified dataset.
- Save checkpoints in `./checkpoint/`.
- Print final evaluation metrics:

  - Precision
  - Recall
  - F1-score
  - Overall and per-class AUC

## Project Structure

```
BYOL-ACL/
├── pretrained/           # Pretrained BYOL ResNet-18 encoders
├── helpers/
│   ├── models.py         # Classifier and encoder loading utilities
│   ├── training.py       # Training and evaluation functions
│   ├── utils.py          # Helper functions (e.g., weight loading)
│   └── data.py           # Dataset loading and splitting
│
├── byol_pretrain.py      # Main BYOL pre-training script
├── train.py              # Main training and evaluation script
├── requirements.txt
└── README.md
```

## Citation

TBA
