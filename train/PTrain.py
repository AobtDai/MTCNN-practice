import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PDataset import PDataset
from torch.utils.data import DataLoader
import argparse
import yaml
from easydict import EasyDict
import sys
sys.path.append("..")
from model.ConvModels import PNet, LossFn


parser = argparse.ArgumentParser(description='PNet of MTCNN')
parser.add_argument("--config_path", type=str, default="../config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["MTCNN"]

P_anno_train_path = config["Train"]["P_annotation_train_path"]
P_anno_val_path = config["Val"]["P_annotation_val_path"]
batch_size = config["Train"]["batch_size"]
epochs = config["Train"]["epochs"]
lr = config["Train"]["lr"]
P_is_load = config["Train"]["P_is_load"]
P_load_path = config["Train"]["P_load_path"]
P_save_path = config["Train"]["P_save_path"]


def eval(model):
    print(" -------< Evaluating >------- ")
    model.eval()
    eval_dataset = PDataset(annotation_path = P_anno_val_path)
    eval_loader = DataLoader(dataset = eval_dataset, 
                              shuffle = True, 
                              batch_size = batch_size,
                              drop_last = False)
    acc_num = 0
    tot_num = 0 
    for i, batch in enumerate(eval_loader):
        img_tensor, label_tensor, offest_tensor = batch # torch.tensor
        with torch.no_grad():
            # pred_tensor = model(img_tensor.to(device))
            cls_pred_tensor, bbox_pred_tensor = model(img_tensor.to(device))
            pred_tensor = torch.squeeze(cls_pred_tensor)
            gt_tensor = torch.squeeze(label_tensor)
            for j in range(gt_tensor.size(0)):
                if gt_tensor[j]==-1:
                    continue
                if gt_tensor[j]==1:
                    if pred_tensor[j]>0.6:
                        acc_num+=1
                else:
                    if pred_tensor[j]<0.6:
                        acc_num+=1
                tot_num+=1

    acc_rate = acc_num / tot_num
    print("Eval: Acc_rate %.2f%% in %d \n" % (acc_rate*100, tot_num))
    model.train()
    return acc_rate


if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = PNet()
    model.to(device)
    model.train()

    if P_is_load:
        print(" -------< Loading parameters from {} >------- \n".format(P_load_path))
        params = torch.load(P_load_path, map_location='cuda:0')
        model.load_state_dict(params, strict=True) 

    loss_function = LossFn() 
    optimizer = optim.Adam(model.parameters(), lr = lr)
    train_dataset = PDataset(annotation_path = P_anno_train_path)
    train_loader = DataLoader(dataset = train_dataset, 
                              shuffle = True, 
                              batch_size = batch_size,
                              drop_last = True)
    best_acc = 0.988
    epoch_record_x = []
    acc_rate_record_y = []
    epoch_record_x.append(0)
    acc_rate_y = eval(model)
    acc_rate_record_y.append(acc_rate_y)

    for epoch in range(0, epochs+1):
        for i, batch in enumerate(train_loader):
            img_tensor, label_tensor, offest_tensor = batch # torch.tensor
            cls_pred_tensor, bbox_pred_tensor = model(img_tensor.to(device))
            cls_loss = loss_function.cls_loss(cls_pred_tensor, label_tensor.to(device))
            bbox_loss = loss_function.bbox_loss(bbox_pred_tensor, label_tensor.to(device), offest_tensor.to(device))
            tot_loss = cls_loss + 0.5*bbox_loss
            # print("Train: \n", cls_loss, "\n", bbox_loss, "\n", tot_loss)
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()
        
        print("Epoch" , epoch)
        epoch_record_x.append(epoch)
        acc_rate_y = eval(model)
        acc_rate_record_y.append(acc_rate_y)
        
        if best_acc < acc_rate_y:
            print(" -------< Saved Best Model >------- \n")
            torch.save(model.state_dict(), P_save_path)
            best_acc = acc_rate_y

        if epoch in [3,5]:
            lr = optimizer.param_groups[0]['lr']
            print(" -------< Reducing lr to {} >------- \n".format(lr/2))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr/2
        
    
    plt.plot(epoch_record_x, acc_rate_record_y, 
                color="green", label="acc_rate_record")
    plt.title("Acc with epoch")
    plt.legend()
    plt.show()
