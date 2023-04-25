r""" this file is to generate both newly annotation file """

import os
import torch
import argparse
import yaml
from easydict import EasyDict
import numpy as np
import cv2
import sys
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append("..")
from utils import GetIoU_t, GetNMS
from model.ConvModels import PNet
from train.PDataset import PDatasetDetect


parser = argparse.ArgumentParser(description='MTCNN RNet Data Annotation Preparing')
parser.add_argument("--config_path", type=str, default="../config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
anno_config = config["MTCNN"]["Anno"]
train_config = config["MTCNN"]["Train"]

anno_train_path = anno_config["anno_train_path"]
image_path_prefix = anno_config["image_path_prefix"]
rnet_anno_path = anno_config["rnet_anno_path"]
rnet_posimg_path = anno_config["rnet_posimg_path"]
rnet_parimg_path = anno_config["rnet_parimg_path"]
rnet_negimg_path = anno_config["rnet_negimg_path"]
pos_num = 0
neg_num = 0
par_num = 0
RNet_data_file = open(rnet_anno_path, "w")
P_model_path = train_config["P_load_path"]
# P_anno_train_path = train_config["P_annotation_train_path"]


def GenerateBBox(cls_tensor, bbox_tensor, scale):
    ksize = 12 # pnet likes a 12*12 kernel
    stride = 2 # multiple all the strides in pnet
    x, y = torch.where(cls_tensor > 0.6) # 0.6 is threshold
    if x.shape[0] == 0:
        return torch.tensor([])
    dx1, dy1, dx2, dy2 = [bbox_tensor[i, x, y] for i in range(4)]
    # bbox_tensor = torch.stack([dx1, dy1, dx2, dy2])
    # print(bbox_tensor.shape) #[4, ?]
    score = cls_tensor[x, y]
    # print(score.shape)
    bbox = torch.stack([torch.round((stride * x)/scale), 
                        torch.round((stride * y)/scale),
                        torch.round((stride * x + ksize)/scale),
                        torch.round((stride * y + ksize)/scale),
                        score,
                        # bbox_tensor,
                        dx1, dy1, dx2, dy2,
                        ])
    return bbox.T


def draw_test(path, boxes):
    # boxes = boxes[0]
    img = cv2.imread(path)
    for box in boxes:
        # print(box)
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
    cv2.imshow("test", img)
    cv2.waitKey(0)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pnet = PNet()
    print(" -------< Loading parameters from {} >------- \n".format(P_model_path))
    params = torch.load(P_model_path, map_location='cuda:0')
    pnet.load_state_dict(params, strict=True) 
    pnet.to(device)
    pnet.eval()

    detect_dataset = PDatasetDetect(annotation_path = anno_train_path,
                                    prefix_path = image_path_prefix)
    detect_loader = DataLoader(dataset = detect_dataset, 
                              shuffle = False, 
                              batch_size = 1,
                              drop_last = False)
    
    # final_boxes = []
    for index, batch in enumerate(detect_loader):
        img_tensor, p, gtboxes = batch # torch.tensor
        # print(gtboxes.shape)
        p = p[0] #####
        gtboxes = gtboxes[0].to(device)
        if gtboxes.shape[0]==0:
            continue
        tot_boxes = []
        _, c, h, w = img_tensor.shape
        ksize = 12 # pnet likes a 12*12 kernel
        iscale = 1
        ih, iw = h, w
        # ih&iw are for iteration

        while min(ih, iw) > ksize:
            with torch.no_grad():
                cls_pred_tensor, bbox_pred_tensor = pnet(img_tensor.to(device))
                cls_pred_tensor = cls_pred_tensor[0][0]
                bbox_pred_tensor = bbox_pred_tensor[0]
            boxes = GenerateBBox(cls_pred_tensor, bbox_pred_tensor, iscale)
            ih *= 0.7
            iw *= 0.7
            iscale = int(min(ih, iw))*1.0 / min(h, w)
            resize_trans = transforms.Resize(int(min(ih, iw)))
            img_tensor = resize_trans(img_tensor[0])
            _, ih, iw = img_tensor.shape
            img_tensor = torch.unsqueeze(img_tensor, dim=0)
            if boxes.shape[0] == 0:
                continue
            valid_index = GetNMS(boxes, 0.5, "Union") 
            boxes = boxes[valid_index]
            tot_boxes.append(boxes)

        if len(tot_boxes) == 0:
            # final_boxes.append(torch.tensor([]))
            continue

        tot_boxes = torch.vstack(tot_boxes)
        valid_index = GetNMS(tot_boxes, 0.7, "Union")
        tot_boxes = tot_boxes[valid_index]
        boxw = tot_boxes[:, 2] - tot_boxes[:, 0] + 1
        boxh = tot_boxes[:, 3] - tot_boxes[:, 1] + 1
        # boxes = torch.stack([tot_boxes[:, 0],
        #                      tot_boxes[:, 1],
        #                      tot_boxes[:, 2],
        #                      tot_boxes[:, 3],
        #                      tot_boxes[:, 4],
        #                      ])
        # boxes = boxes.T
        boxes_orig = torch.stack([tot_boxes[:, 0] + tot_boxes[:, 5] * boxw,
                                  tot_boxes[:, 1] + tot_boxes[:, 6] * boxh,
                                  tot_boxes[:, 2] + tot_boxes[:, 7] * boxw,
                                  tot_boxes[:, 3] + tot_boxes[:, 8] * boxh,
                                  tot_boxes[:, 4]
                                  ])
        boxes_orig = boxes_orig.T
        valid_index = [True for _ in range(boxes_orig.shape[0])]   
        for i in range(boxes_orig.shape[0]):
            if boxes_orig[i][2]-boxes_orig[i][0] <= 3 or boxes_orig[i][3]-boxes_orig[i][1] <= 3:
                valid_index[i] = False
                # print('pnet has one smaller than 3')
            else:
                if boxes_orig[i][2] < 1 or boxes_orig[i][0] > w-2 or boxes_orig[i][3] < 1 or boxes_orig[i][1] > h-2:
                    valid_index[i] = False
                    # print('pnet has one out')

        # final_boxes.append(boxes_orig[valid_index, :])
        raw_boxes = boxes_orig[valid_index, :]
        raw_img = cv2.imread(p)
        for box in raw_boxes:
            x1, y1, x2, y2, _ = box.int()
            x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
            # print(x1, y1, x2, y2)
            bw = x2 - x1 + 1
            bh = y2 - y1 + 1
            if min(bw, bh)<20 or x1<0 or y1<0 or x2>w-1 or y2>h-1:
                continue

            iou_tensor = GetIoU_t(box, gtboxes)
            iou = torch.max(iou_tensor).item()
            crop_img = raw_img[y1:y2, x1:x2, : ]
            new_img = cv2.resize(crop_img, (24, 24), interpolation=cv2.INTER_LINEAR)

            if iou < 0.3 and neg_num<60: # neg #####neg_num necessary?
                save_path = os.path.join(rnet_negimg_path, "%d.jpg"%neg_num)
                RNet_data_file.write(save_path + " 0\n")
                cv2.imwrite(save_path, new_img)
                neg_num += 1
                if neg_num%100==0:
                    print("%d neg_img has been generated"%neg_num)
            else:
                idx = torch.argmax(iou_tensor).item()
                gtx1, gty1, gtx2, gty2 = gtboxes[idx]
                gtx1, gty1, gtx2, gty2 = gtx1.item(), gty1.item(), gtx2.item(), gty2.item()

                # compute bbox reg label
                offx1 = (gtx1 - x1) / float(w)
                offy1 = (gty1 - y1) / float(h)
                offx2 = (gtx2 - x2) / float(w)
                offy2 = (gty2 - y2) / float(h)

                if iou > 0.65: # pos
                    save_path = os.path.join(rnet_posimg_path, "%d.jpg"%pos_num)
                    RNet_data_file.write(save_path + " 1 %.2f %.2f %.2f %.2f\n"
                                        %(offx1, offy1, offx2, offy2))
                    cv2.imwrite(save_path, new_img)
                    pos_num += 1
                    if pos_num%100==0:
                        print("%d pos_img has been generated"%pos_num)

                elif iou > 0.4: # par
                    save_path = os.path.join(rnet_parimg_path, "%d.jpg"%par_num)
                    RNet_data_file.write(save_path + " -1 %.2f %.2f %.2f %.2f\n"
                                        %(offx1, offy1, offx2, offy2))
                    cv2.imwrite(save_path, new_img)
                    par_num += 1
                    if par_num%100==0:
                        print("%d par_img has been generated"%par_num)


        # draw_test(p, raw_boxes)
        # boxes = boxes[valid_index, :]
        # aaaaa = torch.where()

RNet_data_file.close()

