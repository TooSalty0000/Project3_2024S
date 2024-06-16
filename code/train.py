import os
import os.path as osp
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from datasets import Proj3_Dataset
from models import *

np.random.seed(240604)


def get_args_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lr", default="1e-3")
    argparser.add_argument("--optim_type", default="adam")
    argparser.add_argument("--arch_ver", default="ver13")
    argparser.add_argument("--freeze", action="store_true")
    argparser.add_argument("--ver_name", default="")
    args = argparser.parse_args()
    return args


def split_trainval(num_train=45, num_val=10):
    trainval_annos = pd.read_csv("datasets/train_anno.csv")

    categories = sorted(trainval_annos["cls"].unique())
    train_annos, val_annos = [], []
    for c in categories:
        idxs = np.arange(num_train + num_val)
        np.random.shuffle(idxs)
        tgt_df = trainval_annos.groupby("cls").get_group(c).reset_index(drop=True)
        train_annos.append(tgt_df.loc[idxs[:num_train]])
        val_annos.append(tgt_df.loc[idxs[num_train:]])

    train_annos = pd.concat(train_annos).reset_index(drop=True)
    val_annos = pd.concat(val_annos).reset_index(drop=True)
    return train_annos, val_annos


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)
    return images, labels


def stack_crops(crops):
    return torch.stack([T.ToTensor()(crop) for crop in crops])


def run_val_epoch(net, data_loader):
    net.eval()
    sum_loss = 0
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for idx, (img, gt_y) in enumerate(data_loader):
            img, gt_y = img.to(device), gt_y.to(device)
            img = (img - img_mean) / img_std

            if img.dim() == 5:
                batch_size, num_crops, C, H, W = img.shape
                img = img.view(
                    -1, C, H, W
                )  # reshape to [batch_size * num_crops, C, H, W]
                gt_y = gt_y.repeat(num_crops)
                out = net(img)
                out = out.view(batch_size, num_crops, -1).mean(1)  # average over crops
            else:
                batch_size = img.size(0)
                out = net(img)

            _, pred = torch.max(out, 1)
            correct += pred.eq(gt_y[:batch_size].data).sum().item()

            loss = criterion(out, gt_y[:batch_size])
            sum_loss += gt_y.size(0) * loss.item()
            num_samples += gt_y.size(0)

    loss = sum_loss / num_samples
    acc = 100 * correct / num_samples

    return loss, acc


def run_trainval():
    ep = -1
    val_loss, val_acc = run_val_epoch(net, val_loader)
    print(f"[val-{ep + 1}/{num_epochs}] loss: {val_loss:.6f} | acc: {val_acc:.3f}%")
    writer.add_scalar("ep_loss/val", val_loss, ep + 1)
    writer.add_scalar("ep_acc/val", val_acc, ep + 1)

    for ep in range(num_epochs):
        net.train()
        ep_loss = 0
        ep_pred_y, ep_gt_y = [], []
        start_time = datetime.now()

        for idx, (img, gt_y) in enumerate(tqdm(train_loader)):
            img, gt_y = img.to(device), gt_y.to(device)
            img = (img - img_mean) / img_std

            if img.dim() == 5:
                batch_size, num_crops, C, H, W = img.shape
                img = img.view(
                    -1, C, H, W
                )  # reshape to [batch_size * num_crops, C, H, W]
                gt_y = gt_y.repeat(num_crops)
                pred_y = net(img)
                pred_y = pred_y.view(batch_size, num_crops, -1).mean(
                    1
                )  # average over crops
            else:
                batch_size = img.size(0)
                pred_y = net(img)

            loss = criterion(pred_y, gt_y[:batch_size])
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += len(gt_y[:batch_size]) * loss.item()

            ep_pred_y.append(pred_y.detach().max(dim=1)[1].cpu())
            ep_gt_y.append(gt_y[:batch_size].cpu())

        # Adjust learning rate using scheduler
        scheduler.step()

        # Save checkpoints
        if (ep + 1) % save_intv == 0 or (ep + 1) == num_epochs:
            torch.save(net.state_dict(), osp.join(ckpt_dir, f"ep{ep+1}.pt"))

        end_time = datetime.now()
        print(f"Time elapsed {end_time - start_time}")

        ep_pred_y = torch.cat(ep_pred_y)
        ep_gt_y = torch.cat(ep_gt_y)
        train_loss = ep_loss / len(ep_gt_y)
        train_acc = 100 * (ep_gt_y == ep_pred_y).to(float).mean().item()
        val_loss, val_acc = run_val_epoch(net, val_loader)

        print(
            f"[train-{ep + 1}/{num_epochs}] loss: {train_loss:.6f} | acc: {train_acc:.3f}%"
        )
        print(f"[val-{ep + 1}/{num_epochs}] loss: {val_loss:.6f} | acc: {val_acc:.3f}%")
        writer.add_scalar("ep_loss/train", train_loss, ep + 1)
        writer.add_scalar("ep_loss/val", val_loss, ep + 1)
        writer.add_scalar("ep_acc/train", train_acc, ep + 1)
        writer.add_scalar("ep_acc/val", val_acc, ep + 1)

        # Implement early stopping if needed
        if val_acc >= 94:
            print(f"Early stopping at epoch {ep + 1}")
            torch.save(net.state_dict(), osp.join(ckpt_dir, f"ep{ep+1}.pt"))
            break


if __name__ == "__main__":
    args = get_args_parser()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 200
    save_intv = 5
    lr = float(args.lr)
    weight_decay = 1e-4
    num_workers = 2
    batch_size = 64
    freeze_backbone = args.freeze
    num_cls = 50
    optim_type = args.optim_type
    arch_ver = args.arch_ver
    output_dir = f'outputs/arch{arch_ver}_lr{lr}_freeze{"T" if freeze_backbone else "F"}_optim{optim_type}'
    if args.ver_name != "":
        output_dir += f"_V{args.ver_name}"
    ckpt_dir = osp.join(output_dir, "ckpt")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)

    optim_choices = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    model_choices = {
        "ver1": R34_ver1,
        "ver2": R34_ver2,
        "ver3": R34_ver3,
        "ver4": R34_ver4,
        "ver5": R34_ver5,
        "ver6": R34_ver6,
        "ver7": R34_ver7,
        "ver8": R34_ver8,
        "ver9": R34_ver9,
        "ver10": R34_ver10,
        "ver11": R34_ver11,
        "ver12": R34_ver12,
        "ver13": R34_ver13,
        "ver14": R34_ver14,
        "ver15": R34_ver15,
    }

    # Split train/val datasets randomly - you can modify this randomness
    train_annos, val_annos = split_trainval(num_train=45, num_val=10)

    # Data transform
    img_size = 256
    crop_size = 224
    max_rotation = 50
    train_transform = T.Compose(
        [
            T.Resize(img_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(max_rotation),
            T.RandomResizedCrop(crop_size, scale=(0.7, 1.2)),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.2),
            T.ToTensor(),
            T.RandomErasing(),
        ]
    )
    val_transform = T.Compose(
        [
            T.Resize(img_size),
            T.FiveCrop(crop_size),
            T.Lambda(stack_crops),
        ]
    )

    # Build dataloader
    train_dataset = Proj3_Dataset(
        train_annos, "train", train_transform, num_augmentations=1
    )
    val_dataset = Proj3_Dataset(val_annos, "val", val_transform)

    print("Train dataset: #", len(train_dataset))
    print("Val dataset: #", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=custom_collate_fn,
    )

    # Build model
    net = model_choices[arch_ver](num_cls=num_cls, freeze_backbone=freeze_backbone).to(
        device
    )

    # Train & validation
    img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
    img_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)
    criterion = nn.CrossEntropyLoss()  # Loss function - you can define others
    train_parameters = filter(lambda p: p.requires_grad, net.parameters())
    optim = optim_choices[optim_type](
        train_parameters, lr=lr, weight_decay=weight_decay
    )
    scheduler = StepLR(optim, step_size=10, gamma=0.1)

    run_trainval()
