# python frame_cla.py --RESUME False --EPOCH 10 --model_path full_apc_sll.pth --model_name apc
# python frame_cla.py --RESUME False --EPOCH 10 --model_path ./model_parameter/truncated2/final_40context_model.pth --model_name 40context

import kaldiark
import numpy as np
from classifier_net import FrameClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import cv2
import os
import random
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--RESUME", help="Whether to continue", default='True', type=str)
parser.add_argument("--EPOCH", help="Number of epochs", default=10, type=int)
parser.add_argument("--model_path", help="Model path", type=str)
parser.add_argument("--model_name", help="Model name", type=str)
config = parser.parse_args()


# RESUME = False
# context_length = 40
k = 3 # time shift
# EPOCH = 15
in_dim = 40

pretrain_path = config.model_path
save_check_path = './model_parameter/class/check_classifier_{}.pth'.format(config.model_name)
save_model_path = '.classifier_{}.pth'.format(config.model_name)

data_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.fbank.scp'
dev_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-dev.fbank.scp'
bpali_scp_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.bpali.scp'
bpali_path = '/home/htang2/proj/data/wsj/ext-data/train-si284.bpali'
norm_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.mean-var'

with open('/home/htang2/proj/data/wsj/ext-data/phones-unk.txt', 'r') as phone_file:
    phones = phone_file.readlines()
    phone_list = [p.strip() for p in phones]
phone2id = {n: i for i, n in enumerate(phone_list)} 
idx2phone = {i: w for i, w in enumerate(phone_list)}
n_class = len(phone2id)

bp_dict = {}
with open(bpali_scp_path, 'r') as bpali_scp:  
    bpali_lines = bpali_scp.readlines()
    for line in bpali_lines:
        file_index = line.split(' ')[0].strip()
        bp_index = line.split(':')[1].strip()
        bp_dict.setdefault(file_index, bp_index) # {'011c0201': '9',

net = FrameClassifier(n_class, pretrain_path).to(device)
print(net)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
        
print("RESUME = ", config.RESUME)
print("Model path: ", config.model_path)
print("Model name: ", config.model_name)
start_epoch = -1
if config.RESUME == 'True':
    path_checkpoint = save_check_path  
    checkpoint = torch.load(path_checkpoint)  
    net.load_state_dict(checkpoint['net'])  
    optimizer.load_state_dict(checkpoint['optimizer'])  
    start_epoch = checkpoint['epoch']  
    # lr_schedule.load_state_dict(checkpoint['lr_schedule'])



# classifier model
# net = FrameClassifier(n_class, pretrain_path).to(device)
# print(net)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)



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

print('Start training')
bpali_txt = open(bpali_path, 'r')
for epoch in range(start_epoch + 1,config.EPOCH):
    optimizer.zero_grad()
    optimizer.step()
    running_loss = 0.0
    print("Epoch: ", epoch)
    count = 0 
    correct = 0
    label_num = 0
    data_file = open(data_path, 'r')
    data_lines = data_file.readlines()
    for line in data_lines:
        count += 1
        file_index = line.split(' ')[0].strip() # 011c0201
        line = line.split(' ')[1].strip()
        fbank_path = line.split(':')[0]    #U
        fbank_index = line.split(':')[1]
        
        # utter
        f = open(fbank_path, 'rb')
        f.seek(int(fbank_index))
        mat = kaldiark.parse_feat_matrix(f)
        mat = (mat - mean) / std

        mat = np.expand_dims(mat, axis=0)
        mat = torch.Tensor(mat).to(device)

        # labels
        bpali_txt.seek(int(bp_dict[file_index]))
        bpali_txt_lines = bpali_txt.readline().strip()
        phones = bpali_txt_lines.split(' ')
        labels = [phone2id[p] for p in phones]
        label_num += len(labels)
        labels = torch.LongTensor(labels).to(device)
#         print(labels)
        
        pred = net(mat)
        optimizer.zero_grad()
        loss = criterion(pred, labels)
#         print(pred)
#         print('pred_shape',pred.shape)
        _, pred_max = torch.max(pred, 1)
#         print('grad_max', pred_max)
#         print('correct: ', (pred_max == labels).sum().item())
        correct += (pred_max == labels).sum().item()
        acc = (pred_max == labels).sum().item()
        acc = acc / len(labels)
        
        
        # train
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if count % 100 == 0:
            print("Model:{}, Epoch:{}, count:{}, loss:{}, acc:{}".format(config.model_path, epoch, count, loss.item(), acc))

    
    accuracy = correct / label_num
    fer_error = 1 - accuracy
    print('Accuracy is: ', accuracy)
    print('Frame error rate: ', fer_error)
    
    # Save checkpoint
    checkpoint = {
        "net": net.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
#             'lr_schedule': lr_schedule.state_dict()
    }
 
    torch.save(net.state_dict(), './model_parameter/class/{}_e{}.pth'.format(config.model_name, epoch))
    torch.save(checkpoint, './model_parameter/class/ckpt_{}_e{}.pth'.format(config.model_name, epoch))
    torch.save(checkpoint, save_check_path)
        
    mean_loss = running_loss/count
    print("loss in this epoch: ", mean_loss)
    data_file.close()
    

print("final loss is: ", mean_loss)
print("Finish training.")
torch.save(net.state_dict(), save_model_path)