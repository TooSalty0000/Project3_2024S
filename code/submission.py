import os
import os.path as osp
import numpy as np
import pandas as pd
import argparse

import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import torchvision.transforms as T
import einops

from datasets import Proj3_Dataset
from models import *

np.random.seed(240604)

def get_args_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--arch_ver', default='ver1')
    argparser.add_argument('--num_crop', default=1, type=int, choices=[1, 3, 5, 10])
    argparser.add_argument('--ckpt_path', required=True)
    args = argparser.parse_args()
    return args

def stack_crops(crops):
    return torch.stack([T.ToTensor()(crop) for crop in crops])

def run_eval(net, data_loader):
    net.eval()

    output_logit = []
    with torch.no_grad():
        for idx, img in enumerate(data_loader):
            num_batch = img.shape[0]
            if multi_crop_flag:
                img = einops.rearrange(img, 'b v c h w -> (b v) c h w', b=num_batch)

            img = img.to(device)
            img = (img - img_mean) / img_std
            out = net(img)
            prob = out.softmax(dim=1)
            if multi_crop_flag:
                prob = einops.rearrange(prob, '(b v) nc -> b v nc', b=num_batch).mean(dim=1)

            output_logit.append(prob.cpu().numpy())

    output_logit = np.concatenate(output_logit)
    output_cls = np.argmax(output_logit, axis=1)

    return output_cls

if __name__ == '__main__':
    '''
    python submission.py --arch_ver {arch_ver} --ckpt_path outputs/{path_to_your_model}/ckpt/ep30.pt --num_crop {1/3/5/10} 
    python3 submission.py --arch_ver ver14 --ckpt_path outputs/archver14_lr5e-05_freezeF_optimadam_Vtest1/ckpt/ep10.pt --num_crop 1
    python3 submission.py --arch_ver ver15 --ckpt_path outputs/archver15_lr5e-05_freezeF_optimadam_Vtest2/ckpt/ep20.pt --num_crop 5
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--arch_ver', default='ver1')
    argparser.add_argument('--num_crop', default=1, type=int, choices=[1, 3, 5, 10])
    argparser.add_argument('--ckpt_path', required=True)
    args = argparser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 50
    num_workers = 2
    batch_size = 8
    num_cls = 50
    arch_ver = args.arch_ver
    model_choices = {
        'ver1': R34_ver1, 
        'ver2': R34_ver2, 
        'ver3': R34_ver3, 
        'ver4': R34_ver4, 
        'ver5': R34_ver5, 
        'ver6': R34_ver6, 
        'ver7': R34_ver7,
        'ver8': R34_ver8,
        'ver9': R34_ver9,
        'ver10': R34_ver10,
        'ver11': R34_ver11,
        'ver12': R34_ver12,
        'ver13': R34_ver13,
        'ver14': R34_ver14,
        'ver15': R34_ver15,
    }
    crop_choices = {1: T.CenterCrop, 5: T.FiveCrop, 10: T.TenCrop}
    img_size = 256
    crop_size = 224
    multi_crop_flag = True if args.num_crop > 1 else False

    ## build dataset
    test_subm = pd.read_csv('datasets/test_subm.csv')
    val_transform = T.Compose([
        T.Resize(img_size),
        crop_choices[args.num_crop](crop_size),
        T.Lambda(stack_crops) if multi_crop_flag else T.ToTensor(),
    ])

    test_subm_dataset = Proj3_Dataset(test_subm, 'test', val_transform)
    test_subm_loader = DataLoader(test_subm_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    print("Test dataset: #", len(test_subm_dataset))

    ## build and load model
    net = model_choices[arch_ver](num_cls=num_cls).to(device)
    net.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device(device)))
    summary(net, input_size=(3, crop_size, crop_size))

    ## forward
    img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
    img_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)
    output_cls = run_eval(net, test_subm_loader)

    ## save as csv file
    SID = 3
    test_subm['cls'] = output_cls
    test_subm.to_csv(f'datasets/{SID}_test_subm.csv', index=False)
