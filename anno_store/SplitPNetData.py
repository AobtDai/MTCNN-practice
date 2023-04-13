r""" this file is to generate both newly annotation file """

import os
import random
import argparse
import yaml
from easydict import EasyDict
import numpy as np

parser = argparse.ArgumentParser(description='MTCNN PNet Data Annotation Preparing')
parser.add_argument("--config_path", type=str, default="../config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["MTCNN"]["Anno"]

pnet_anno_path = config["pnet_anno_path"]
pnet_anno_train_path = config["pnet_anno_train_path"]
pnet_anno_val_path = config["pnet_anno_val_path"]
# PNet_data_file = open(pnet_anno_path, "r")
train_file = open(pnet_anno_train_path, "w")
val_file = open(pnet_anno_val_path, "w")


if __name__ == "__main__":
    num = [0, 0, 0]
    with open(pnet_anno_path, "r") as f:
        annotations = f.readlines()

    for annotation in annotations:
        s = annotation.strip(" ").split(" ")
        kind = int(s[1])
        num[kind] += 1
        if num[kind]==10:
            num[kind]=0
            val_file.write(annotation)
        else :
            train_file.write(annotation)


val_file.close()
train_file.close()