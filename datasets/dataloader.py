import os
import io
import re
import cv2
import random
import numpy as np
from PIL import Image
from typing import Optional
from pprint import pprint

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class TemporalRescale(object):
    def __init__(self, temp_scaling=0.2):
        self.min_len = 32
        self.max_len = 230
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        new_len = max(self.min_len, min(new_len, self.max_len))
        
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4

        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))

        return [clip[i] for i in index]


class VideoDataset(Dataset):
    def __init__(self, dataset_name="phoenix14", split="train", num_frames=150):
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

        self.num_frames = num_frames
        self.split = split

        self.features_paths = {
            "phoenix14": "/data/sharedData/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px",
        }

        self.db_info = {
            "phoenix14": "/home/g202302610/Code/OpFlow-SLR/datasets/phoenix2014",
        }

        self.features_path = self.features_paths[dataset_name]
        self.db_info_path = self.db_info[dataset_name]
        
        self.gloss_dict_path = os.path.join(self.db_info_path, "gloss_dict.npy")
        self.gloss_dict = np.load(self.gloss_dict_path, allow_pickle=True).item()

        self.i2g_dict = dict((v[0], k) for k, v in self.gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}

        self.info_path = os.path.join(self.db_info_path, f"{split}_info.npy")
        self.temp_data = np.load(self.info_path, allow_pickle=True).item()
        self.temp_data.pop("prefix")

        self.data = []
        self.labels = []
        for k in self.temp_data.keys():
            path = self.temp_data[k]["folder"].split("/")[:-1]
            path = os.path.join(self.features_path, "/".join(path))
            label = self.temp_data[k]["label"]
            self.data.append(path)

            label = label.split(" ")
            label = [gloss for gloss in label if gloss not in ["", " "]]

            label = [self.g2i_dict[gloss] for gloss in label]
            label = torch.tensor(label)
            self.labels.append(label)

        c = 0
        non_existent = []
        for path in self.data:
            if not os.path.exists(path):
                non_existent.append(path)
                c+=1

        for path in non_existent:
            idx = self.data.index(path)
            self.data.pop(idx)
            self.labels.pop(idx)

        print("removed non-existent paths:", c)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        self.temp_rescale = TemporalRescale(temp_scaling=0.2)
        self.color_aug = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),])
        self.hor_flip = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)])
        self.affine_transform = transforms.Compose([transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)])
        self.rot = transforms.Compose([transforms.RandomRotation(degrees=(0, 10))])
        self.gaussian_blur = transforms.Compose([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data[idx]
        label = self.labels[idx]

        frame_files = sorted(os.listdir(video_path), key=numerical_sort_key)
        frame_files = [os.path.join(video_path, f) for f in frame_files]

        indices = np.linspace(0, len(frame_files) - 1, min(len(frame_files), self.num_frames), dtype=int)
        selected_frames = [frame_files[i] for i in indices]

        frames = [self.transform(Image.open(frame)) for frame in selected_frames]

        if self.split == "train":
            frames = self.temp_rescale(frames)
            if random.random() < 0.5: frames = [self.color_aug(frame) for frame in frames]
            if random.random() < 0.5: frames = [self.hor_flip(frame) for frame in frames]
            if random.random() < 0.5: frames = [self.affine_transform(frame) for frame in frames]
            if random.random() < 0.5: frames = [self.gaussian_blur(frame) for frame in frames]
            if random.random() < 0.5: frames = [self.rot(frame) for frame in frames]

        return frames, label, len(label), video_path


def numerical_sort_key(filename):
    match = re.search(r'fn(\d+)', filename)
    return int(match.group(1)) if match else float('inf') 


def batch_padder(batch):
    max_len = max(len(frames) for frames, _, _, _ in batch)

    if max_len % 4 != 0:
        max_len += 4 - (max_len % 4)

    padded_batch = []
    labels = []
    label_lgt = []
    vid_paths = []

    for frames, label, lgt, video_path in batch:
        pad_needed = max_len - len(frames)
        pad_start = pad_needed // 2
        pad_end = pad_needed - pad_start
        padded_frames = [frames[0]] * pad_start + frames + [frames[-1]] * pad_end
        padded_batch.append(torch.stack(padded_frames))

        label_lgt.append(lgt)
        labels.extend(label)

        vid_paths.append(video_path)

    padded_batch = torch.stack(padded_batch)
    label_lgt = torch.LongTensor(label_lgt)
    labels = torch.stack(labels)

    return padded_batch, labels, label_lgt, vid_paths


def construct_loader(dataset_name, split="train", logger=None):
    dataset = VideoDataset(dataset_name=dataset_name, split=split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        collate_fn=batch_padder
    )

    return loader


if __name__ == "__main__":
    dataset = VideoDataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        # shuffle=True,
        drop_last=True,
        collate_fn=batch_padder
    )

    for frames, label, label_lgt in loader:
        pass

