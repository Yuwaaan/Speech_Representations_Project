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
parser.add_argument("--EPOCH", help="Number of epochs", default=15, type=int)
parser.add_argument("--model_path", help="Model path", type=int)
parser.add_argument("--model_name", help="Model name", type=int)
config = parser.parse_args()


# RESUME = False
# context_length = 40
k = 3 # time shift
# EPOCH = 15
in_dim = 40

pretrain_path = config.model_path
save_check_path = './model_parameter/truncated/check_classifier_{}.pth'.format(config.model_name)
save_model_path = '.classifier_{}.pth'.format(config.model_name)

data_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.fbank.scp'
dev_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-dev.fbank.scp'
bpali_scp_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.bpali.scp'
bpali_path = '/home/htang2/proj/data/wsj/ext-data/train-si284.bpali'


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
net = FrameClassifier(n_class, pretrain_path).to(device)
print(net)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


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


print('Start training')
for epoch in range(start_epoch + 1,EPOCH):
    running_loss = 0.0
    for i, (file_index, mat, labels) in enumerate(train_dataloader, start=0):
        # mat.size = [1, 652, 40]
        pred = net(mat)
        optimizer.zero_grad()
        loss = criterion(pred, labels) * 40
        _, pred_max = torch.max(pred, 1)
        correct += (pred_max == labels).sum().item()
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 300 == 0:
            print("Epoch:{} count:{} loss:{}".format(epoch, i, loss.item()))
            
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
    if not os.path.isdir("./model_parameter/truncated"):
        os.makedirs("./model_parameter/truncated")

    torch.save(checkpoint, save_check_path)
        
    mean_loss = running_loss/len(train_dataloader)
    print("loss in this epoch: ", mean_loss)
    

print("final loss is: ", mean_loss)
print("Finish training.")
torch.save(net.state_dict(), save_model_path)



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
        
 