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

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", help="Model path", type=str)
config = parser.parse_args()

m = 3   # time shift


net = APCModel().to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


# model_path = './full_20context.pth' # ***

dev_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-dev.fbank.scp'
# bpali_scp_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-dev.bpali.scp'
# bpali_path = '/home/htang2/proj/data/wsj/ext-data/dev93.bpali'   
norm_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.mean-var'

def norm_frame(norm_path):
    norm_file = open(norm_path)
    line1 = np.array(eval(norm_file.readline()))
    line2 = np.array(eval(norm_file.readline()))
    num_samples = int(norm_file.readline())
    mean = line1 / num_samples
    stddev = np.sqrt(line2 / num_samples - mean * mean)
    norm_file.close()
    return mean, stddev
mean, std = norm_frame(norm_path)


def dev_model(dev_path, model_path):
    dev_file = open(dev_path, 'r')
    dev_lines = dev_file.readlines()
    dev_loss = 0
    dev_count = 0
    with torch.no_grad():
        for line in dev_lines:
            dev_count += 1
            file_index = line.split(' ')[0].strip() # 011c0201
            line = line.split(' ')[1].strip()

            fbank_path = line.split(':')[0]
            fbank_index = line.split(':')[1]
#             print("count:{} utterance index:{}".format(dev_count, fbank_index))

            # utter
            f = open(fbank_path, 'rb')
            f.seek(int(fbank_index))
            mat = kaldiark.parse_feat_matrix(f)
            mat = (mat - mean) / std

            mat = np.expand_dims(mat, axis=0)
            mat = torch.Tensor(mat).to(device)

            out = net(mat)
            loss = criterion(out[:,:-m,:], mat[:,m:,:]) * 40
            print("model:{}, count:{}, file_index:{}, loss:{}".format(model_path, dev_count, file_index[0], loss.item()))
            dev_loss += loss.item()
            
    dev_file.close()
    dev_loss = dev_loss / dev_count
    return dev_loss


net.load_state_dict(torch.load(config.model_path))
print('dev model ..')
dev_loss = dev_model(dev_path,config.model_path)
print("Dev loss: ", dev_loss)
