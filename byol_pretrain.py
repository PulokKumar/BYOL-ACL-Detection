import argparse

import os
from helpers.models import BYOLNet, get_resnet18_encoder_random
from helpers.training import train_byol

import torch
import random
import numpy as np

from helpers.data import get_unl_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_root_unl", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()


def main():

    seed = args.seed
    data_root_unl = args.data_root_unl

    byol_epochs = args.epochs
    byol_lr = 2e-4
    byol_weight_decay = 1e-6
    byol_momentum_base = 0.996
    byol_batch_size = 256

    save_dir = "./pretrained"
    os.makedirs(save_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    unl_dataset = get_unl_dataset(data_root_unl)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_online = get_resnet18_encoder_random()
    encoder_target = get_resnet18_encoder_random()

    online_byol = BYOLNet(encoder_online, encoder_dim=512)
    target_byol = BYOLNet(encoder_target, encoder_dim=512)

    train_byol(
        online_byol, target_byol,
        unl_dataset,
        epochs=byol_epochs,
        batch_size=byol_batch_size,
        lr=byol_lr,
        weight_decay=byol_weight_decay,
        momentum_base=byol_momentum_base,
        save_dir=save_dir,
        device=device,
        seed=seed,
    )


if __name__ == '__main__':
    main()
