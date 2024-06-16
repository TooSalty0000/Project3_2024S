import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import Proj3_Dataset

np.random.seed(419030)


def show_images(images, titles=None):
    """Display a list of images"""
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(15, 15))
    for i, img in enumerate(images):
        img = img.permute(1, 2, 0)  # change from (C, H, W) to (H, W, C)
        axs[i].imshow(img)
        axs[i].axis("off")
        if titles is not None:
            axs[i].set_title(titles[i])
    plt.show()


def main():
    # Read annotations
    annos = pd.read_csv("datasets/train_anno.csv")

    # Data transformation
    img_size = 256
    crop_size = 224
    max_rotation = 30

    transform = T.Compose(
        [
            T.Resize(img_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(max_rotation),
            T.RandomResizedCrop(crop_size, scale=(0.4, 1.0)),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.2),
            T.ToTensor(),
            T.RandomErasing(),
        ]
    )

    # Create dataset
    dataset = Proj3_Dataset(annos, "train", transform)

    # Load a few samples
    loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2)

    # Display augmented images
    for idx, (imgs, labels) in enumerate(loader):
        if idx == 10:  # Just display one batch
            break
        show_images(imgs, titles=[f"Class: {label.item()}" for label in labels])


if __name__ == "__main__":
    main()
