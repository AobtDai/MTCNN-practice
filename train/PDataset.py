import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch

class PDataset(Dataset):
    def __init__(self, annotation_path):
        super(PDataset, self).__init__()
        self.img_paths = []
        self.img_labels = []
        self.img_offsets = []
        with open(annotation_path, "r") as f:
            annotations = f.readlines()
        self.sample_num = len(annotations)

        for annotation in annotations:
            splited_string = annotation.strip(" ").split(" ")
            img_path = splited_string[0]
            img_label = int(splited_string[1])
            self.img_paths.append(img_path)
            self.img_labels.append(img_label)
            img_offset = torch.zeros(4)
            if img_label:
                img_offset[0] = float(splited_string[2])
                img_offset[1] = float(splited_string[3])
                img_offset[2] = float(splited_string[4])
                img_offset[3] = float(splited_string[5])
            
            self.img_offsets.append(img_offset)

    def __getitem__(self, index):
        # img = Image.open(self.img_paths[index]).convert('L')
        img = Image.open(self.img_paths[index])
        trans = transforms.ToTensor()
        # trans = transforms.Compose([transforms.ToTensor()])
        img = trans(img)
        label = self.img_labels[index]
        offset = self.img_offsets[index]
        return img, label, offset
    
    def __len__(self):
        return self.sample_num
    
    