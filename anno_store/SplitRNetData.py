r""" this file is to generate both newly annotation file """
r""" for there are some problems, this file turn to special uses"""

import os
import random
import argparse
import yaml
from easydict import EasyDict
import numpy as np

parser = argparse.ArgumentParser(description='MTCNN RNet Data Annotation Preparing')
parser.add_argument("--config_path", type=str, default="../config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["MTCNN"]["Anno"]

rnet_anno_path = config["rnet_anno_path"]
rnet_anno_train_path = config["rnet_anno_train_path"]
rnet_anno_val_path = config["rnet_anno_val_path"]
# PNet_data_file = open(pnet_anno_path, "r")
train_file = open(rnet_anno_train_path, "w")
val_file = open(rnet_anno_val_path, "w")

# neg_num = 26410
# par_num = 83042
# pos_num = 27714
# prefix_path = r"""C:\Users\25705\Downloads\documents\fdu\1.6\CV\PJ\Face-Recog\mtcnn\processed_data\RNet"""

if __name__ == "__main__":
    num = [0, 0, 0]
    # num = 0
    with open(rnet_anno_path, "r") as f:
        annotations = f.readlines()

    for annotation in annotations:
        s = annotation.strip(" ").split(" ")
        kind = int(s[1])
    # for i in range(0, neg_num):
        num[kind] += 1
        if num[kind]==10:
            num[kind]=0
            val_file.write(annotation)
        else :
            train_file.write(annotation)


val_file.close()
train_file.close()