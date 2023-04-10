import kaldiark
import numpy as np
from apc import APCModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import cv2
import os
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", help="Model path", type=int)
config = parser.parse_args()

# path
dev_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-dev.fbank.scp'
bpali_scp_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.bpali.scp'
bpali_path = '/home/htang2/proj/data/wsj/ext-data/train-si284.bpali'

# classifier model
net = FrameClassifier(n_class, pretrain_path).to(device)
print(net)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


# dev model
def dev_model(dev_dataloader):
    running_loss = 0.0
    with torch.no_grad():
        for i, (file_index, mat, labels) in enumerate(dev_dataloader, start=0):
            # Train model
            pred = net(mat)
            loss = criterion(pred, labels) * 40
            _, pred_max = torch.max(pred, 1)
            correct += (pred_max == labels).sum().item()
            sample_loss = loss.item() 

            if i % 100 == 0:
                print("Regular APC, Epoch:{}, count:{}, file_index:{}, loss:{}".format(epoch, i, file_index[0], sample_loss))
            running_loss += sample_loss
        mean_loss = running_loss / len(dev_dataloader)
        accuracy = correct / label_num
        fer_error = 1 - accuracy
        print('Accuracy is: ', accuracy)
        print('Frame error rate: ', fer_error)
        return mean_loss

net.load_state_dict(torch.load(save_model_path))

print('Loading dev data ..')
devset = WSJDataset(dev_path, bpali_path)
dev_dataloader = torch.utils.data.DataLoader(devset, batch_size=1, shuffle=True, num_workers=0)

print('dev model ..')
dev_loss = dev_model(dev_path)
print("Dev loss: ", dev_loss)
