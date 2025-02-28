import os
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import argparse
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from utils.logger import Logger
from models.model import load_config, build_model



def main(args):
    config = load_config(args.config)

    print(config)



if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', dest='config')
    args=parser.parse_args()

    main(args)