import numpy as np

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