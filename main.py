import os
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import argparse
import datetime
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from pprint import pprint

from utils.logger import Logger
from utils.optimizer import Optimizer
from models.model import load_config, build_model

from models.criterions import SeqKD
from datasets.dataloader import construct_loader
from utils.write2file import write2file
from utils.wer_evaluation import wer_calculation


def train_epoch(model, optimizer, train_loader, device, config, args, logger):
    ctc_fun = nn.CTCLoss(reduction='none', zero_infinity=False)
    dist_fun = SeqKD(T=8)

    loss_value = []
    model.train()
    for frames, labels, label_lgt, path in tqdm(train_loader):
        optimizer.zero_grad()

        frames = frames.to(device)
        for label in labels:
            label = label.to(device)

        vid_lgt = torch.tensor([len(frame) for frame in frames])

        out = model(frames, vid_lgt)
        loss = ctc_fun(
            out["conv_logits"].log_softmax(-1), 
            labels,
            out["feat_len"].long(), 
            label_lgt
        ).mean()

        loss += ctc_fun(
            out["sequence_logits"].log_softmax(-1),
            labels,
            out["feat_len"].long(), 
            label_lgt
        ).mean()

        loss += 10 * dist_fun(
            out["conv_logits"].log_softmax(-1),
            out["sequence_logits"].detach(),
            use_blank=False
        )

        loss.backward()
        optimizer.step()
        loss_value.append(loss.item())
            
    optimizer.scheduler.step()
    return np.mean(loss_value)

        
def validation(model, val_loader, device, output_dir, config, args, logger, epoch, mode="dev"):
    model.eval()
    with torch.no_grad():
        total_sent, total_info = [], []

        for frames, labels, label_lgt, paths in tqdm(val_loader):
            frames = frames.to(device)
            for label in labels:
                label = label.to(device)

            vid_lgt = torch.tensor([len(frame) for frame in frames])
            out = model(frames, vid_lgt)

            total_sent += out["recognized_sents"]
            total_info += [p.split("/")[-2] for p in paths]

        write2file(os.path.join(output_dir, f"output-hypothesis-{mode}-{epoch}.ctm"), total_info, total_sent)

        return wer_calculation(
            os.path.join(config["data"]["dataset_root"], f"phoenix2014-groundtruth-{mode}-{epoch}.stm"),
            os.path.join(output_dir, f"output-hypothesis-{mode}.ctm")
        )

def save_model(model, optimizer, epoch, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': optimizer.scheduler.state_dict(),
    }, save_path)

def main(args):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)

    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config_name = args.config.split("/")[-1].split(".")[0]

    output_dir = f"./outputs/{config_name}/{date}/"
    if not os.path.exists(f"./outputs/{config_name}/{date}"):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "dev_outputs"))
        os.makedirs(os.path.join(output_dir, "models"))
        
    logger = Logger(f"{output_dir}/log.txt")

    train_loader = construct_loader(config["data"]["dataset_name"], "train", logger)
    val_loader = construct_loader(config["data"]["dataset_name"], "dev", logger)
    test_loader = construct_loader(config["data"]["dataset_name"], "test", logger)

    model = build_model(config, logger).to(device)
    optimizer = Optimizer(model, config["optimizer_args"])

    for k,v in model.named_parameters():
        if v.requires_grad:
            logger(k, verbose=False)
    
    logger("\n")

    best_wer = 100000
    if args.mode == "train":
        for epoch in range(config["training"]["epochs"]):
            train_loss = train_epoch(model, optimizer, train_loader, device, config, args, logger)
            eval_wer = validation(model, val_loader, device, output_dir, config, args, logger, epoch)

            logger('\tMean training loss: {:.10f}.'.format(train_loss))
            logger(f"Validation WER: {eval_wer}")

            if eval_wer < best_wer:
                best_wer = eval_wer
                save_model(model, optimizer, epoch, f"{output_dir}/models/best_model.pth")

    logger("\n")

    model.load_state_dict(torch.load(f"{output_dir}/models/best_model.pth")['model_state_dict'])
    eval_wer = validation(model, test_loader, device, output_dir, config, args, logger, 0, mode="test")
    logger(f"Test WER: {eval_wer}")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', dest='config')
    parser.add_argument('--mode', dest='mode', default='train')

    args=parser.parse_args()

    main(args)