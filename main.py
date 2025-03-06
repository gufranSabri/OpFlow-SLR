import os
import torch
import torch.nn.utils
import random
import argparse
import datetime
from tqdm import tqdm
import numpy as np
import torch.nn as nn

from utils.logger import Logger
from models.model import load_config, build_model

from models.criterions import SeqKD
from utils.write2file import write2file
from utils.wer_evaluation import wer_calculation
from utils.scheduler import LinearDecayLR

from datasets.dataloader_video import *

from utils.device import GpuDataParallel
from utils.optimizer import Optimizer

from models.sync_batchnorm import convert_model

#TODO: Add all modules to swin model
#TODO: Add contrastive learning
#TODO: Implement SlowFast model

def construct_loader(split="train"):
    gloss_dict_path = os.path.join("/home/g202302610/Code/OpFlow-SLR/datasets/phoenix2014", "gloss_dict.npy")
    gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()
    dataset_args = {
        "mode": 'train',
        "datatype": 'video',
        "num_gloss": -1,
        "prefix": "/data/sharedData/phoenix2014-release/phoenix-2014-multisigner",
        "mode": split,
        "transform_mode": split == "train",
    }
    dataset = BaseFeeder(gloss_dict=gloss_dict, **dataset_args)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=2 if split == "train" else 8,
        shuffle=False,
        drop_last=split == "train",
        num_workers=10,
        collate_fn=dataset.collate_fn,
    )

def train_epoch(model, optimizer, train_loader, device, config, logger):
    ctc_fun = nn.CTCLoss(reduction='none', zero_infinity=False)
    dist_fun = SeqKD(T=8)

    loss_value = []
    model.train()
    loss_log_interval = 0
    for i, data in enumerate(tqdm(train_loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])

        ret_dict = model(vid, vid_lgt)
        loss_conv = ctc_fun(
            ret_dict["conv_logits"].log_softmax(-1), 
            label.cpu().int(),
            ret_dict["feat_len"].cpu().int(), 
            label_lgt.cpu().int()
        ).mean()

        loss_main = ctc_fun(
            ret_dict["sequence_logits"].log_softmax(-1),
            label.cpu().int(),
            ret_dict["feat_len"].cpu().int(), 
            label_lgt.cpu().int()
        ).mean()

        loss_kd = 25 * dist_fun(
            ret_dict["conv_logits"],
            ret_dict["sequence_logits"].detach(),
            use_blank=False
        )

        loss = loss_conv + loss_main + loss_kd

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            tqdm.write(f"NANNED on {data[-1]}")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        loss_value.append(loss.item())

        loss_log_interval += 1
        if loss_log_interval % 50 == 0:
            logger(f"Batch Loss [{loss_log_interval-50}-{loss_log_interval}]: {np.mean(loss_value[-10:])}")
            
    return np.mean(loss_value)

def validation(model, val_loader, device, output_dir, config, logger, epoch, mode="dev"):
    model.eval()
    with torch.no_grad():
        total_sent, total_info = [], []

        for i, data in enumerate(tqdm(val_loader)):
            vid = device.data_to_device(data[0])
            vid_lgt = device.data_to_device(data[1])

            ret_dict = model(vid, vid_lgt)
            total_info += [file_name.split("|")[0] for file_name in data[-1]]
            total_sent += ret_dict['recognized_sents']

        write2file(os.path.join(output_dir, f"output-hypothesis-{mode}-{epoch}.ctm"), total_info, total_sent)

        return wer_calculation(
            os.path.join(config["data"]["dataset_root"], f"phoenix2014-groundtruth-{mode}.stm"),
            os.path.join(output_dir, f"output-hypothesis-{mode}-{epoch}.ctm")
        )

def save_model(model, optimizer, epoch, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

def model_to_device(model, device):
    model = model.to(device.output_device)
    if len(device.gpu_list) > 1:
        model.visual = nn.DataParallel(
            model.visual,
            device_ids=device.gpu_list,
            output_device=device.output_device
        )
    model = convert_model(model)
    model.cuda()
    return model

def main(args):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    config = load_config(args.config)

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config_name = args.config.split("/")[-1].split(".")[0]

    output_dir = f"./outputs/{config_name}/{date}/"
    if not os.path.exists("./outputs"): os.makedirs("./outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "models"))
        
    logger = Logger(f"{output_dir}/log.txt")

    train_loader = construct_loader("train")
    val_loader = construct_loader("dev")
    test_loader = construct_loader("test")

    model = build_model(config, logger)
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=config["optimizer_args"]["base_lr"],
    #     weight_decay=config["optimizer_args"]["weight_decay"],
    # )
    # lr_scheduler=LinearDecayLR(optimizer, config["training"]["epochs"], 25)
    optimizer = Optimizer(model, config["optimizer_args"])


    device = GpuDataParallel()
    device.set_device(config["training"]["device"])
    model = model_to_device(model, device)

    best_wer = 10000000
    if args.mode == "train":
        for epoch in range(config["training"]["epochs"]):
            logger(f"Epoch [{epoch + 1}/{config['training']['epochs']}]")
            train_loss = train_epoch(model, optimizer, train_loader, device, config, logger)
            eval_wer = validation(model, val_loader, device, output_dir, config, logger, epoch)

            # logger(f"Learning Rate: {lr_scheduler.get_lr()}\n")
            logger('\tMean training loss: {:.10f}.'.format(train_loss))
            logger(f"\tValidation WER: {eval_wer}")
            logger("\n")

            if eval_wer < best_wer:
                best_wer = eval_wer
                save_model(model, optimizer, epoch, f"{output_dir}/models/best_model.pth")

            # lr_scheduler.step()
            optimizer.scheduler.step()

    logger("\n")

    model.load_state_dict(torch.load(f"{output_dir}/models/best_model.pth")['model_state_dict'])
    eval_wer = validation(model, test_loader, device, output_dir, config, logger, 0, mode="test")
    logger(f"Test WER: {eval_wer}")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', dest='config')
    parser.add_argument('--mode', dest='mode', default='train')

    args=parser.parse_args()

    main(args)