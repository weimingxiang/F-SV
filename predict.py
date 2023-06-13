import os
import random
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pysam
import pytorch_lightning as pl
import ray
import torch
import torch.nn as nn
import torchvision
from hyperopt import hp
from pudb import set_trace
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback, TuneReportCheckpointCallback)
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest import Repeater
from ray.tune.suggest.hyperopt import HyperOptSearch
import pandas as pd
import re
import list2img
import utilities as ut
from net import IDENet

num_cuda = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = num_cuda

# data_dir = "../datasets/NA12878_PacBio_MtSinai/"
data_dir = "/home/xwm/DeepSVFilter/datasets/NA12878_PacBio_MtSinai/"

bam_path = data_dir + "sorted_final_merged.bam"


# get chr list
sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224

# pool = Pool(2)
for chromosome, chr_len in zip(chr_list, chr_length):
    if(chromosome == "chr1"):
        break

print("======= deal " + chromosome + " =======")

# chromosome_sign
if os.path.exists(data_dir + "chromosome_sign/" + chromosome + ".pt"):
    print("========== loading ==========")
    chromosome_sign = torch.load(
        data_dir + "chromosome_sign/" + chromosome + ".pt")
    mid_sign = torch.load(
        data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt")
    mid_sign_img = torch.load(
        data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign.pt")
    print("========== loaded ==========")

else:
    ut.mymkdir(data_dir + "chromosome_sign/")
    chromosome_sign, mid_sign, mid_sign_list = ut.preprocess(
        bam_path, chromosome, chr_len, data_dir)
    torch.save(chromosome_sign, data_dir +
               "chromosome_sign/" + chromosome + ".pt")
    torch.save(mid_sign, data_dir + "chromosome_sign/" +
               chromosome + "_mids_sign.pt")
    torch.save(mid_sign_list, data_dir +
               "chromosome_sign/" + chromosome + "_m(i)d_sign.pt")
    # mid_sign_img = ut.mid_list2img(mid_sign_list, chromosome)
    mid_sign_img = torch.tensor(list2img.deal_list(mid_sign_list))
    ut.mymkdir(data_dir + "chromosome_img/")
    torch.save(mid_sign_img, data_dir +
               "chromosome_img/" + chromosome + "_m(i)d_sign.pt")

resize = torchvision.transforms.Resize([512, 11])

config = {
    "batch_size": 14,
    "beta1": 0.9,
    "beta2": 0.999,
    "lr": 7.187267009530772e-06,
    "weight_decay": 0.0011614665567890423,
    "model_name": "resnet50",
    "KFold": 5,
    "KFold_num": 0,
    #     "classfication_dim_stride": 20, # no use
}

# model = IDENet.load_from_checkpoint(
#     "/home/xwm/DeepSVFilter/code/checkpoints_predict/7+11channel_predict_5fold/4-epoch=15-validation_f1=0.92-validation_mean=0.92.ckpt", path=data_dir, config=config)
model = IDENet.load_from_checkpoint(
    "/home/xwm/DeepSVFilter/code/checkpoints_predict/7+11channel_predict_5fold/4-epoch=15-validation_f1=0.94-validation_mean=0.94.ckpt", path=data_dir, config=config)

model = model.eval().cuda()


# insert
filename = data_dir + "sniffles/output.vcf"
# /home/xwm/DeepSVFilter/datasets/NA12878_PacBio_MtSinai/pbsv/ref.chr1.var.vcf
# /home/xwm/DeepSVFilter/datasets/NA12878_PacBio_MtSinai/svim/variants.vcf
# /home/xwm/DeepSVFilter/datasets/NA12878_PacBio_MtSinai/sniffles/output.vcf
file = open(filename + "_filter94.vcf", 'w')

with open(filename, "r") as f:
    lines = f.readlines()
    for data in lines:
        print(data)
        if "#" in data:
            file.writelines(data)
        else:
            if "DEL" in data:
                data_list = data.split("\t")
                pos_begin = int(data_list[1])
                s = data_list[7]
                if("END" in s):
                    pos = s.find("END") + 4  # "END="
                    s = s[pos:]
                    s = s.split(";")[0]
                    s = int(s)
                else:
                    pos = s.find("SVLEN") + 6
                    s = s[pos:]
                    s = s.split(";")[0]
                    s = pos_begin + int(s)
                end = s
                gap = int((end - pos_begin) / 4)
                if gap == 0:
                    gap = 1
                # positive
                begin = pos_begin - 1 - gap
                end = end - 1 + gap
            elif "INS" in data:
                data_list = data.split("\t")
                pos_begin = int(data_list[1])
                gap = 112
                begin = pos_begin - 1 - gap
                end = pos_begin - 1 + gap
            if begin < 0:
                begin = 0
            if end >= chr_len:
                end = chr_len - 1

            ins_img = ut.to_input_image_single(
                chromosome_sign[:, begin:end])  # dim 3
            ins_img_mid = ut.to_input_image_single(
                mid_sign[:, begin:end])  # dim 4
            ins_img_i = resize(mid_sign_img[begin:end].unsqueeze(0))
            all_ins_img = torch.cat([ins_img, ins_img_mid], 0)
            y_hat = model(all_ins_img.unsqueeze(
                dim=0).cuda(), ins_img_i.cuda())
            type = torch.argmax(y_hat, dim=1)
            if(type == 0):
                pass
            elif(type == 1):  # del
                file.writelines(data.replace('INS', 'DEL'))
            elif(type == 2):  # ins
                file.writelines(data.replace('DEL', 'INS'))


file.close()
