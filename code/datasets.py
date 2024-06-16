from torch.utils.data import Dataset
from torchvision import transforms
import os.path as osp
from PIL import Image
import cv2
import torch


class Proj3_Dataset(Dataset):
    def __init__(self, annos, split, transform=None, num_augmentations=0):
        """
        annos: DataFrame([filename, cls])
        split: train | val | test
        transform: Transformations to apply to the images
        num_augmentations: Number of augmentations per image
        """
        self.annos = annos
        self.transform = transform
        self.split = split
        self.is_test = split == "test"
        self.num_augmentations = num_augmentations

    def __len__(self):
        return len(self.annos) * (self.num_augmentations + 1)

    def __getitem__(self, idx):
        original_idx = idx // (self.num_augmentations + 1)
        augment_idx = idx % (self.num_augmentations + 1)

        tgt_row = self.annos.loc[original_idx]
        filename = tgt_row["filename"]
        filepath = f"datasets/images/{filename}"
        if not osp.isfile(filepath):
            raise Exception(f"{filepath} does not exist")

        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)  # RGB
        img = Image.fromarray(img)

        if self.num_augmentations == 0:
            if self.transform:
                transformed_img = self.transform(img)
            else:
                transformed_img = transforms.Resize((256, 256))(img)
                transformed_img = transforms.CenterCrop((224, 224))(transformed_img)
                transformed_img = transforms.ToTensor()(transformed_img)
        elif augment_idx == 0:
            # Return the original image resized to the desired dimensions
            transformed_img = transforms.Resize((256, 256))(img)
            transformed_img = transforms.CenterCrop((224, 224))(transformed_img)
            transformed_img = transforms.ToTensor()(transformed_img)
        else:
            # Return an augmented image resized to the desired dimensions
            if self.transform:
                img = transforms.Resize((256, 256))(img)
                transformed_img = self.transform(img)
            else:
                transformed_img = transforms.Resize((256, 256))(img)
                transformed_img = transforms.CenterCrop((224, 224))(transformed_img)
                transformed_img = transforms.ToTensor()(transformed_img)

        if self.is_test:
            return transformed_img
        else:
            label = torch.tensor(tgt_row["cls"]).long()
            return transformed_img, label
