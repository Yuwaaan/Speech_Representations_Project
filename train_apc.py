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
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# APC
RESUME = True
k = 3  # time shift
EPOCH = 30


net = APCModel().to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=0.0004)

save_check_path = './model_parameter/truncated/fullapc/check_apc_sll.pth'
save_model_path = './full_apc_sll.pth'

data_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.fbank.scp'
dev_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-dev.fbank.scp'
bpali_scp_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.bpali.scp'
bpali_path = '/home/htang2/proj/data/wsj/ext-data/train-si284.bpali'   



start_epoch = -1
if RESUME:
    path_checkpoint = save_check_path
    checkpoint = torch.load(path_checkpoint)  
    net.load_state_dict(checkpoint['net'])  
    optimizer.load_state_dict(checkpoint['optimizer'])  
    start_epoch = checkpoint['epoch']  
    # lr_schedule.load_state_dict(checkpoint['lr_schedule'])

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15,20], gamma=0.1)   
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)


# Load data
print('Loading data..')
trainset = WSJDataset(data_path, bpali_path)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

print('Start training.')
for epoch in range(start_epoch + 1,EPOCH):
    running_loss = 0.0
    for i, (file_index, mat, labels) in enumerate(train_dataloader, start=0):
        # Train model
        out = net(mat)
        optimizer.zero_grad()
        loss = criterion(out[:,:-k,:], mat[:,k:,:]) * 40       
        loss.backward()
        optimizer.step()
        sample_loss = loss.item()
        
        if i % 100 == 0:
            print("APCsl{}, Epoch:{}, count:{}, file_index:{}, loss:{}".format(k, epoch, i, file_index[0], sample_loss))
        running_loss += sample_loss
        
#     scheduler.step()

    # Save checkpoint
    checkpoint = {
        "net": net.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
#             'lr_schedule': lr_schedule.state_dict()
    }
    if not os.path.isdir("./model_parameter/truncated"):
        os.makedirs("./model_parameter/truncated")
    torch.save(checkpoint, './model_parameter/truncated/fullapc/ckpt_apcsll_%s.pth' % (str(epoch)))
    torch.save(checkpoint, save_check_path)
    torch.save(net.state_dict(), './full_apc_e%s.pth' % (str(epoch)))
    
    mean_loss = running_loss/len(train_dataloader)
    print("loss in this epoch: ", mean_loss)
    save_model_path = './full_apc_sll.pth'

print("final loss is: ", mean_loss)
print("Finish training.")
torch.save(net.state_dict(), save_model_path)