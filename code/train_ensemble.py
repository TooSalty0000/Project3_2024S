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

from datasets import Proj3_Dataset
from models import *

np.random.seed(202407)


def get_args_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lr", default="1e-3")
    argparser.add_argument("--optim_type", default="adam")
    argparser.add_argument("--arch_ver", default="ver1")
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


def run_val_epoch(nets, data_loader, weights):
    sum_loss = 0
    correct = 0
    num_samples = 0
    all_outputs = []

    with torch.no_grad():
        for idx, (img, gt_y) in enumerate(data_loader):
            batch_size, num_crops, C, H, W = img.shape
            img = img.view(-1, C, H, W)  # reshape to [batch_size * 5, C, H, W]
            gt_y = gt_y.repeat(num_crops)

            img, gt_y = img.to(device), gt_y.to(device)
            img = (img - img_mean) / img_std

            outputs = []
            for net in nets:
                out = net(img)
                out = out.view(batch_size, num_crops, -1).mean(1)  # average over crops
                outputs.append(out)

            outputs = torch.stack(outputs)  # shape: (num_models, batch_size, num_cls)
            weighted_sum = torch.zeros_like(outputs[0])
            for i in range(len(nets)):
                weighted_sum += weights[i] * outputs[i]

            pred = torch.argmax(weighted_sum, dim=1)
            correct += pred.eq(gt_y[:batch_size].data).sum().item()

            loss = criterion(weighted_sum, gt_y[:batch_size])
            sum_loss += batch_size * loss.item()
            num_samples += batch_size
            all_outputs.append(weighted_sum.cpu().numpy())

    loss = sum_loss / num_samples
    acc = 100 * correct / num_samples
    all_outputs = np.vstack(all_outputs)

    return loss, acc, all_outputs


def run_trainval():
    ep = -1
    val_loss, val_acc, _ = run_val_epoch(nets, val_loader, weights)
    print(f"[val-{ep + 1}/{num_epochs}] loss: {val_loss:.6f} | acc: {val_acc:.3f}%")
    writer.add_scalar("ep_loss/val", val_loss, ep + 1)
    writer.add_scalar("ep_acc/val", val_acc, ep + 1)

    for ep in range(num_epochs):
        ep_loss = 0
        ep_pred_y, ep_gt_y = [], []
        start_time = datetime.now()

        for idx, (img, gt_y) in enumerate(tqdm(train_loader)):
            img, gt_y = img.to(device), gt_y.to(device)
            img = (img - img_mean) / img_std

            for net, optim in zip(nets, optimizers):
                pred_y = net(img)
                loss = criterion(pred_y, gt_y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                ep_loss += len(gt_y) * loss.item()

                ep_pred_y.append(pred_y.detach().max(dim=1)[1].cpu())
                ep_gt_y.append(gt_y.cpu())

        # Update weights
        weight_optimizer.zero_grad()
        weight_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for net in nets:
            val_loss, val_acc, _ = run_val_epoch(
                [net], val_loader, [torch.ones_like(weights[0])]
            )
            weight_loss = weight_loss + val_loss
        weight_loss.backward()
        weight_optimizer.step()

        end_time = datetime.now()
        print(f"Time elapsed {end_time - start_time}")

        ep_pred_y = torch.cat(ep_pred_y)
        ep_gt_y = torch.cat(ep_gt_y)
        train_loss = ep_loss / len(ep_gt_y)
        train_acc = 100 * (ep_gt_y == ep_pred_y).to(float).mean().item()
        val_loss, val_acc, _ = run_val_epoch(nets, val_loader, weights)

        print(
            f"[train-{ep + 1}/{num_epochs}] loss: {train_loss:.6f} | acc: {train_acc:.3f}%"
        )
        print(f"[val-{ep + 1}/{num_epochs}] loss: {val_loss:.6f} | acc: {val_acc:.3f}%")
        writer.add_scalar("ep_loss/train", train_loss, ep + 1)
        writer.add_scalar("ep_loss/val", val_loss, ep + 1)
        writer.add_scalar("ep_acc/train", train_acc, ep + 1)
        writer.add_scalar("ep_acc/val", val_acc, ep + 1)

        # Save checkpoints
        if (ep + 1) % save_intv == 0 or (ep + 1) == num_epochs:
            for i, net in enumerate(nets):
                ckpt_path = osp.join(ckpt_dir, f"net_{i}_ep{ep+1}.pt")
                torch.save(net.state_dict(), ckpt_path)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = get_args_parser()
    seeds = [202407, 200002, 123452, 23415, 6456425]
    learning_rates = [
        1e-3,
        5e-4,
        1e-4,
        5e-5,
        1e-5,
    ]  # Different learning rates for each model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 100
    save_intv = 5
    lr = float(args.lr)
    weight_decay = 1e-4
    num_workers = 2
    batch_size = 100
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
        "ver4": R34_ver4,
        "ver5": R34_ver5,
        "ver6": R34_ver6,
        "ver7": R34_ver7,
    }

    ## split train/val datasets randomly - you can modify this randomness
    train_annos, val_annos = split_trainval(num_train=45, num_val=10)

    ## data transform
    ## For inference, you may use 5-crop (4 corners and center) - T.FiveCrop(img_size)
    img_size = 256
    crop_size = 224
    max_rotation = 30

    train_transform = T.Compose(
        [
            T.Resize(img_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(max_rotation),
            T.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            T.ToTensor(),
            T.RandomErasing(),
        ]
    )
    val_transform = T.Compose(
        [T.Resize(img_size), T.CenterCrop(crop_size), T.ToTensor()]
    )

    ## build dataloader
    train_dataset = Proj3_Dataset(
        train_annos, "train", train_transform, num_augmentations=5
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
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False
    )

    ## build models and optimizers
    nets = []
    optimizers = []
    for i, arch_ver in enumerate(model_choices):
        set_seed(seeds[i])  # Set the seed for each model
        net = model_choices[arch_ver](
            num_cls=num_cls, freeze_backbone=freeze_backbone
        ).to(device)
        optimizer = optim_choices[optim_type](
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=learning_rates[i],  # Use different learning rates for each model
            weight_decay=weight_decay,
        )
        nets.append(net)
        optimizers.append(optimizer)

    ## define learnable weights for soft voting
    weights = torch.nn.Parameter(torch.ones(len(nets), device=device) / len(nets))
    weight_optimizer = torch.optim.Adam([weights], lr=1e-3)

    ## train & validation
    img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
    img_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)
    criterion = nn.CrossEntropyLoss()  ## loss function - you can define others

    run_trainval()
