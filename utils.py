import numpy as np
import torch

def GetIoU(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    
    in_x1 = np.maximum(box[0], boxes[:, 0])
    in_y1 = np.maximum(box[1], boxes[:, 1])
    in_x2 = np.minimum(box[2], boxes[:, 2])
    in_y2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, in_x2 - in_x1 + 1)
    h = np.maximum(0, in_y2 - in_y1 + 1)
    in_area = w * h
    
    iou = in_area / (box_area + area - in_area)
    return np.max(iou)

def GetIoU_t(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    
    in_x1 = torch.maximum(box[0], boxes[:, 0])
    in_y1 = torch.maximum(box[1], boxes[:, 1])
    in_x2 = torch.minimum(box[2], boxes[:, 2])
    in_y2 = torch.minimum(box[3], boxes[:, 3])

    w = torch.maximum(torch.tensor(0.), in_x2 - in_x1 + 1)
    h = torch.maximum(torch.tensor(0.), in_y2 - in_y1 + 1)
    in_area = w * h
    
    iou = in_area / (box_area + area - in_area)
    # return torch.max(iou).item()
    return iou

def GetNMS(boxes, threshold, mode="Union"):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order_index = torch.flip(scores.argsort(),[0])
    valid_index = []

    while order_index.shape[0]>0:
        i = order_index[0]
        valid_index.append(i.item()) ## without item(), it causes indexerror
        in_x1 = torch.maximum(x1[i], x1[order_index[1:]])
        in_y1 = torch.maximum(y1[i], y1[order_index[1:]])
        in_x2 = torch.minimum(x2[i], x2[order_index[1:]])
        in_y2 = torch.minimum(y2[i], y2[order_index[1:]])

        w = torch.maximum(torch.tensor(0.), in_x2 - in_x1 + 1)
        h = torch.maximum(torch.tensor(0.), in_y2 - in_y1 + 1)
        in_area = w * h

        if mode == "Union":
            iou = in_area / (areas[i] + areas[order_index[1:]] - in_area)
        # elif mode == "Minimum":
            # iou = in_area / torch.minimum(areas[i], areas[order_index[1:]])
        order_index = order_index[torch.where(iou<=threshold)[0]+1]
    
    return valid_index