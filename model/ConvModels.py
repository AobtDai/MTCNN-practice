import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # need weight init?
    
    def forward(self, x):
        x = self.pre_layer(x)
        cls = F.sigmoid(self.conv_cls(x))
        bbox_offset = self.conv_bbox(x)
        lmk_offset = self.conv_lmk(x)
        return cls, bbox_offset, lmk_offset
    


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

        cls = F.sigmoid(self.conv_cls(x))
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

        cls = F.sigmoid(self.conv_cls(x))
        bbox = self.conv_bbox(x)
        lmk = self.conv_lmk(x)

        return cls, bbox, lmk