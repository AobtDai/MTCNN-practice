r""" this file is to generate both newly annotation file """

import os
import random
import argparse
import yaml
from easydict import EasyDict
import numpy as np
import cv2
import sys
sys.path.append("..")
from utils import GetIoU


parser = argparse.ArgumentParser(description='MTCNN PNet Data Annotation Preparing')
parser.add_argument("--config_path", type=str, default="../config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["MTCNN"]["Anno"]

anno_train_path = config["anno_train_path"]
image_path_prefix = config["image_path_prefix"]
pnet_anno_path = config["pnet_anno_path"]
pnet_posimg_path = config["pnet_posimg_path"]
pnet_parimg_path = config["pnet_parimg_path"]
pnet_negimg_path = config["pnet_negimg_path"]
pos_num = 0
neg_num = 0
par_num = 0
PNet_data_file = open(pnet_anno_path, "w")

def SaveNewImg(img, crop_box, bboxes, offsets):
    global pos_num
    global neg_num
    global par_num
    # print(neg_num, " ", pos_num, " ", par_num)
    iou = GetIoU(crop_box, bboxes)
    crop_img = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], : ]
    new_img = cv2.resize(crop_img, (12, 12), interpolation=cv2.INTER_LINEAR)
    if iou < 0.3: # neg
        save_path = os.path.join(pnet_negimg_path, "%d.jpg"%neg_num)
        PNet_data_file.write(save_path + " 0\n")
        cv2.imwrite(save_path, new_img)
        neg_num += 1
        if neg_num%100==0:
            print("%d neg_img has been generated"%neg_num)

    elif iou > 0.65 and offsets[0]!="N": # pos
        save_path = os.path.join(pnet_posimg_path, "%d.jpg"%pos_num)
        PNet_data_file.write(save_path + " 1 %.2f %.2f %.2f %.2f\n"
                             %(offsets[0], offsets[1], offsets[2], offsets[3]))
        cv2.imwrite(save_path, new_img)
        pos_num += 1
        if pos_num%100==0:
            print("%d pos_img has been generated"%pos_num)

    elif iou > 0.4 and offsets[0]!="N": # par
        save_path = os.path.join(pnet_parimg_path, "%d.jpg"%par_num)
        PNet_data_file.write(save_path + " -1 %.2f %.2f %.2f %.2f\n"
                             %(offsets[0], offsets[1], offsets[2], offsets[3]))
        cv2.imwrite(save_path, new_img)
        par_num += 1
        if par_num%100==0:
            print("%d par_img has been generated"%par_num)


def GeneratePosParCandidate(img, box, bboxes, exp_num):
    x1, y1, x2, y2 = box
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    for _ in range(0, exp_num):
        img_h, img_w, img_c = img.shape
        offsets = [0., 0., 0., 0.] #
        new_size = np.random.randint(min(box_h, box_w)*0.9, max(box_h, box_w)*1.1) 
        new_y1 = np.random.randint(max(0, y1-new_size/2), min(img_h-new_size/2, y1+new_size/2))
        new_x1 = np.random.randint(max(0, x1-new_size/2), min(img_w-new_size/2, x1+new_size/2))
        crop_box = np.array([new_x1, new_y1, new_x1+new_size, new_y1+new_size])
        offsets[0] = (x1 - new_x1) / float(new_size)
        offsets[1] = (y1 - new_y1) / float(new_size)
        offsets[2] = (x2 - crop_box[2]) / float(new_size)
        offsets[3] = (y2 - crop_box[3]) / float(new_size)
        SaveNewImg(img, crop_box, bboxes, offsets)
    
    new_size = max(box_h, box_w)
    crop_box = np.array([x1, y1, x1+new_size, y1+new_size])
    offsets = [0., 0., 0., 0.] #
    offsets[0] = (x1 - crop_box[0]) / float(new_size)
    offsets[1] = (y1 - crop_box[1]) / float(new_size)
    offsets[2] = (x2 - crop_box[2]) / float(new_size)
    offsets[3] = (y2 - crop_box[3]) / float(new_size)
    SaveNewImg(img, crop_box, bboxes, offsets)

    new_size = min(box_h, box_w)
    crop_box = np.array([x1, y1, x1+new_size, y1+new_size])
    offsets = [0., 0., 0., 0.] #
    offsets[0] = (x1 - crop_box[0]) / float(new_size)
    offsets[1] = (y1 - crop_box[1]) / float(new_size)
    offsets[2] = (x2 - crop_box[2]) / float(new_size)
    offsets[3] = (y2 - crop_box[3]) / float(new_size)
    SaveNewImg(img, crop_box, bboxes, offsets)


def GenerateNegCandidate(img, bboxes, exp_num):
    for _ in range(0, exp_num):
        img_h, img_w, img_c = img.shape
        new_size = np.random.randint(12, min(img_h, img_w)/2) # 12 is the size of pnet input
        new_y1 = np.random.randint(0, img_h-new_size)
        new_x1 = np.random.randint(0, img_w-new_size)
        crop_box = np.array([new_x1, new_y1, new_x1+new_size, new_y1+new_size])
        crop_box = np.array([new_x1, new_y1, new_x1+new_size, new_y1+new_size])
        SaveNewImg(img, crop_box, bboxes, ["N"])


if __name__ == "__main__":

    with open(anno_train_path, "r") as f:
        annotations = f.readlines()
    # iiii = 0 ## debug flag
    for annotation in annotations:
        annotation = annotation.strip(" ").split(" ")
        img_path = os.path.join(image_path_prefix, annotation[0])
        # print(img_path)
        img = cv2.imread(img_path)

        bbox_num = int(annotation[1])
        if bbox_num==0:
            continue
        _bbox = list(map(int, annotation[2:-1]))
        bboxes = np.array(_bbox).reshape(-1, 4)

        # GenerateNegCandidate(img, bboxes, 5)
        for box in bboxes:
            # print(box)
            x1, y1, x2, y2 = box
            box_w = x2 - x1 + 1
            box_h = y2 - y1 + 1
            # if min(box_w, box_h)<12 or max(box_w, box_h)<40:
            if min(box_w, box_h)<12:
                continue
            GenerateNegCandidate(img, bboxes, 5)
            GeneratePosParCandidate(img, box, bboxes, 50)
        # iiii += 1
        # if iiii > 10:
        #     break

PNet_data_file.close()

