import os
from helpers.data import get_weighted_sampler
from helpers.models import Classifier
from helpers.utils import load_encoder_weights_into

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

import copy

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def update_momentum(online_net, target_net, m):
    """
    EMA update: target.params = m * target.params + (1-m) * online.params
    """
    for online_p, target_p in zip(online_net.parameters(), target_net.parameters()):
        target_p.data = target_p.data * m + online_p.data * (1.0 - m)


def byol_loss_fn(p_online, z_target):
    # p_online and z_target are normalized to unit length -> use MSE of vectors
    # equivalent to 2*(1 - cosine_sim) but stable
    return 2 - 2 * (p_online * z_target).sum(dim=-1).mean()


def compute_embedding_variance(encoder, sample_loader, device):
    encoder.eval()
    with torch.no_grad():
        imgs, _ = next(iter(sample_loader))
        imgs = imgs.to(device)
        feats = encoder(imgs)
        feats = feats.view(feats.size(0), -1)
        var_per_dim = feats.var(dim=0)
        return var_per_dim.mean().item()


def train_byol(
    online_net, target_net,
    unl_dataset,
    epochs=50,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-6,
    momentum_base=0.996,
    save_dir="./pretrained",
    device="cpu",
    seed=42,
):
    unl_loader = DataLoader(unl_dataset, batch_size=batch_size, shuffle=False)

    # Init BYOL target network
    online_net.to(device)
    target_net.to(device)
    target_net.load_state_dict(online_net.state_dict())
    for p in target_net.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(online_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    step = 0

    for epoch in range(epochs):
        online_net.train()
        pbar = tqdm(unl_loader, desc=f"BYOL Epoch {epoch+1}/{epochs}")
        for x1, x2 in pbar:
            x1 = x1.to(device)
            x2 = x2.to(device)

            # BYOL forward pass
            p1, _ = online_net(x1)
            p2, _ = online_net(x2)
            with torch.no_grad():
                _, z1 = target_net(x1)
                _, z2 = target_net(x2)

            loss = 0.5 * (byol_loss_fn(p1, z2) + byol_loss_fn(p2, z1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target net via EMA
            current_m = momentum_base
            update_momentum(online_net, target_net, current_m)

            step += 1
            pbar.set_postfix({'loss': float(loss.item())})

        scheduler.step()

    # Save BYOL encoder
    encoder_state = {
        k.replace("encoder.", ""): v.cpu()
        for k, v in online_net.state_dict().items()
        if k.startswith("encoder.")
    }
    ckpt_path = os.path.join(save_dir, f"byol_encoder_{seed}.pth")
    torch.save(encoder_state, ckpt_path)
    print(f"Saved BYOL encoder weights to {ckpt_path}")


def evaluate_classifier(model, dataset, num_classes, batch_size, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0)

    auc_per_class = {}

    auc = roc_auc_score(
        all_targets, np.array(all_probs), multi_class='ovr', average='weighted'
    )

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    for c in range(num_classes):
        auc_per_class[c] = roc_auc_score(
            (all_targets == c).astype(int), all_probs[:, c]
        )

    return prec, rec, f1, auc, auc_per_class


def train_classifier(
        model, train_dataset, val_dataset, num_classes, head_type,
        epochs=10, batch_size=16, lr_head=1e-3,
        lr_enc=1e-4, freeze_encoder=False,
        save_dir="checkpoint",
        seed=42, device="cpu"):

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    sampler = get_weighted_sampler(train_dataset, num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": lr_enc},
        {"params": model.head.parameters(), "lr": lr_head}
    ],
    )

    criterion = nn.CrossEntropyLoss()
    best_val = 0.0
    best_model_w = None

    for epoch in range(epochs):

        model.train()
        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{epochs}")
        losses = []
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix({'loss': np.mean(losses)})

        # eval
        val_prec, val_rec, val_f1, val_auc, val_auc_per_class = evaluate_classifier(
            model, val_dataset, num_classes, batch_size, device)
        print(
            f"Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print("Per-class AUC:")
        for cls_id, cls_auc in val_auc_per_class.items():
            print(f"  Class {cls_id}: {cls_auc:.4f}"
                  if cls_auc is not None else f"  Class {cls_id}: N/A")

        if val_auc > best_val:
            best_val = val_auc

            best_model_w = copy.deepcopy(model.state_dict())
            torch.save(
                best_model_w, os.path.join(
                    save_dir,
                    f"best_classifier_{head_type}_freeze_encoder_{freeze_encoder}_{seed}.pth"))
            print("Saved best classifier checkpoint.")
    return os.path.join(
        save_dir,
        f"best_classifier_{head_type}_freeze_encoder_{freeze_encoder}_{seed}.pth")
