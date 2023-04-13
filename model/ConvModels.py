import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFn():
    def __init__(self):
        self.loss_cls = nn.BCELoss() # binary cross entropy
        # self.loss_cls = nn.CrossEntropyLoss() # binary cross entropy
        self.loss_bbox = nn.MSELoss() # mean square error
    
    def cls_loss(self, pred_tensor, label_tensor):
        # print("Shape1: ",pred_tensor.shape)
        pred_tensor = torch.squeeze(pred_tensor)
        # print("Shape2: ",pred_tensor.shape)
        label_tensor = torch.squeeze(label_tensor)
        # print("Data2: ",label_tensor.data)
        mask_tensor = torch.ge(label_tensor, 0) # abandon par_img
        valid_pred = torch.masked_select(pred_tensor, mask_tensor).float()
        vaild_label = torch.masked_select(label_tensor, mask_tensor).float()
        return self.loss_cls(valid_pred, vaild_label)

    def bbox_loss(self, pred_tensor, label_tensor, offset_tensor):
        pred_tensor = torch.squeeze(pred_tensor)
        label_tensor = torch.squeeze(label_tensor)
        offset_tensor = torch.squeeze(offset_tensor)
        mask_tensor = label_tensor.eq(0).eq(0)
        valid_index = torch.nonzero(mask_tensor)
        valid_index = torch.squeeze(valid_index)
        valid_pred = pred_tensor[valid_index, :]
        valid_offset = offset_tensor[valid_index, :]
        return self.loss_bbox(valid_pred, valid_offset)


class PNet(nn.Module):
    r""" PNet """
    def __init__(self):
        super(PNet, self).__init__()
        
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU()
        )
        self.conv_cls = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv_bbox = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.conv_lmk = nn.Conv2d(32, 10, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.pre_layer(x)
        cls = torch.sigmoid(self.conv_cls(x))
        bbox_offset = self.conv_bbox(x)
        # lmk_offset = self.conv_lmk(x)
        # return cls, bbox_offset, lmk_offset
        return cls, bbox_offset
    


class RNet(nn.Module):
    r""" RNet """
    def __init__(self) :
        super(RNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            # original: kernel_size=3
            nn.PReLU()
        )

        self.pre_linear_layer = nn.Sequential(
            nn.Linear(2*2*64, 128),
            nn.PReLU()
        )

        self.conv_cls = nn.Linear(128, 1)
        self.conv_bbox = nn.Linear(128, 4)
        self.conv_lmk = nn.Linear(128, 10)
        # need weight init?

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1) #???
        x = self.pre_linear_layer(x)

        cls = torch.sigmoid(self.conv_cls(x))
        bbox = self.conv_bbox(x)
        lmk = self.conv_lmk(x)

        return cls, bbox, lmk

    

class ONet(nn.Module):
    r""" ONet """
    def __init__(self) :
        super(ONet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU()
        )

        self.pre_linear_layer = nn.Sequential(
            nn.Linear(2*2*128, 256),
            nn.PReLU()
        )

        self.conv_cls = nn.Linear(256, 1)
        self.conv_bbox = nn.Linear(256, 4)
        self.conv_lmk = nn.Linear(256, 10)

        # need weight init?

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.pre_linear_layer(x)

        cls = torch.sigmoid(self.conv_cls(x))
        bbox = self.conv_bbox(x)
        lmk = self.conv_lmk(x)

        return cls, bbox, lmk