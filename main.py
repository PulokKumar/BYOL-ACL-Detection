import argparse

import os
from helpers.models import Classifier, get_resnet18_encoder_imagenet, get_resnet18_encoder_random
from helpers.training import evaluate_classifier, train_classifier
from helpers.utils import load_encoder_weights_into

import torch
import random
import numpy as np

from helpers.data import get_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
parser.add_argument("--pretrained", choices=["byol", "imagenet", "random"], default="byol")
parser.add_argument("--byol_pretrained_encoder_ckpt", type=str)
parser.add_argument("--freeze", action="store_true")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def main():

    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

    seed = args.seed
    data_root = args.data_root

    num_classes = 3

    epochs = args.epochs
    batch_size = 16
    lr_head = 1e-3
    lr_enc = 1e-4
    encoder_ckpt = args.byol_pretrained_encoder_ckpt
    freeze_encoder = args.freeze

    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset, val_dataset, test_dataset = get_datasets(
        data_root, train_ratio, val_ratio, test_ratio, seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pretrained == "byol":
        encoder = get_resnet18_encoder_random()
        load_encoder_weights_into(encoder, encoder_ckpt)
    elif args.pretrained == "imagenet":
        encoder = get_resnet18_encoder_imagenet()
    else:
        encoder = get_resnet18_encoder_random()

    model = Classifier(encoder, feat_dim=512, num_classes=num_classes,
                       head_type=args.head).to(device)

    train_classifier(
        model, train_dataset, val_dataset, num_classes, args.head,
        epochs, batch_size, lr_head,
        lr_enc, freeze_encoder,
        save_dir,
        seed, device
    )

    test_prec, test_rec, test_f1, test_auc, test_auc_per_class = evaluate_classifier(
        model, test_dataset, num_classes, batch_size, device)

    print(
        f"Test results -> Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.2f}")

    print("Per-class AUC:")
    for cls_id, cls_auc in test_auc_per_class.items():
        print(f"  Class {cls_id}: {cls_auc:.2f}")


if __name__ == '__main__':
    main()
