import faulthandler
import os
import random

import numpy as np
import torch

from Config import CUDA
from Parameters import parse_args
from Solver import Solver


def set_random_seed_and_cuda(opt):
    # Set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
    torch.cuda.set_device("cuda:" + opt.cuda)


if __name__ == "__main__":
    faulthandler.enable()

    opt = parse_args()
    set_random_seed_and_cuda(opt)
    solver = Solver(opt)
    solver.solve()
    # python Main.py --dataset mosi_Dec --num_workers 0 --bert_freeze part --epochs_num 30