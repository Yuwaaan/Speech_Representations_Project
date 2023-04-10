import kaldiark
import numpy as np
from apc import APCModel
from mydata import WSJDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import cv2
import os
import argparse
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--RESUME", help="Whether to continue", default='True', type=str)
parser.add_argument("--EPOCH", help="Number of epochs", default=15, type=int)
parser.add_argument("--context_length", help="Context length of LSTMs", default=40, type=int)
config = parser.parse_args()



# RESUME = False
# context_length = 40
k = 3 # time shift
# EPOCH = 15
in_dim = 40

net = APCModel().to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


save_check_path = './model_parameter/truncated/check_{}context.pth'.format(config.context_length)
save_model_path = './{}context_model.pth'.format(config.context_length)


data_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.fbank.scp'
dev_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-dev.fbank.scp'
bpali_scp_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.bpali.scp'
bpali_path = '/home/htang2/proj/data/wsj/ext-data/train-si284.bpali'   


start_epoch = -1
print("RESUME = ", config.RESUME)
if config.RESUME == 'True':
    path_checkpoint = save_check_path  
    checkpoint = torch.load(path_checkpoint)  
    net.load_state_dict(checkpoint['net'])  
    optimizer.load_state_dict(checkpoint['optimizer'])  
    start_epoch = checkpoint['epoch']  
    # lr_schedule.load_state_dict(checkpoint['lr_schedule'])


def window(mat,context_length, frame_num, in_dim, stride):
    out = F.unfold(mat, kernel_size=(context_length, in_dim),stride=stride)
    out=out.permute(0,2,1)
    out = out.view((-1,frame_num - context_length + 1,context_length, in_dim)) 
    return out   

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

print("RESUME = ", config.RESUME)
print("Context length = ", config.context_length)
# Load data
print('Loading data..')
trainset = WSJDataset(data_path, bpali_path)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

print('Start training')
for epoch in range(start_epoch + 1,config.EPOCH):
    truncated_loss = 0.0
    for i, (file_index, mat, labels) in enumerate(train_dataloader, start=0):
        frame_num = mat.shape[1]   # mat.size = [1, 652, 40]
        mat_ = torch.unsqueeze(mat, dim=0)  # [1, 1, 652, 40]
        mat = torch.squeeze(mat, dim=0)
        tr_mat = window(mat_, config.context_length, frame_num, in_dim, 1)  # [1, 613, 40, 40]
        tr_mat = torch.squeeze(tr_mat, dim=0)  # [613, 40, 40] (batch=T-context+1, context, 40)
        
        # origin mat size: (652,40)  context=40
        out = net(tr_mat)  # (batch=T-context+1, context, 40)
        optimizer.zero_grad()   
        mat = torch.unsqueeze(mat, dim=1)
        loss = criterion(out[:-k,-1:,:], mat[config.context_length+k-1:,:,:]) * 40   # [608, 1, 40] , [608, 1, 40]
        loss.backward()
        optimizer.step()

        sample_loss = loss.item() 
        if i % 100 == 0:
            print("context_length:{} Epoch:{}, count:{}, loss:{}".format(config.context_length, epoch, i, sample_loss))
        truncated_loss += sample_loss

    # Save checkpoint
    checkpoint = {
        "net": net.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
    }
    if not os.path.isdir("./model_parameter/truncated"):
        os.makedirs("./model_parameter/truncated")
    torch.save(checkpoint, './model_parameter/truncated/{}context_epoch{}.pth'.format(config.context_length, epoch))
    torch.save(checkpoint, save_check_path)

    mean_loss = truncated_loss/len(train_dataloader)
    print("loss in this epoch: ", mean_loss)

print("final loss is: ", mean_loss)
print("Finish training.")
torch.save(net.state_dict(), save_model_path)
