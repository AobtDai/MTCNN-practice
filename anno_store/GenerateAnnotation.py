r""" this file is to generate both newly annotation file """

import os
import random
import argparse
import yaml
from easydict import EasyDict

parser = argparse.ArgumentParser(description='Newly MTCNN Annotation File Generation')
parser.add_argument("--config_path", type=str, default="../config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["MTCNN"]["Anno"]

origin_anno_path = config["origin_anno_path"]
anno_train_path = config["anno_train_path"]

with open(origin_anno_path, 'r') as f:
    origins = f.readlines()
train_file = open(anno_train_path, "w")

i = 0
while i < len(origins):
    if origins[i][2] == "-":
        file_name = str(origins[i][:-1]).strip(" ")
        train_file.write(file_name+" ")
        i+=1
        N = int(origins[i])
        train_file.write(str(N)+" ")
        i+=1
        for j in range(0, N):
            bbox = list(map(int, origins[i].split(" ")[:4]))
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = list(map(str, bbox))
            bbox = (" ".join(bbox))
            train_file.write(bbox+" ")
            i+=1
        train_file.write("\n")
    else:
        i+=1

train_file.close()
