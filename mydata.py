from torch.utils.data import Dataset
import kaldiark
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


phone_unk_txt_path = '/home/htang2/proj/data/wsj/ext-data/phones-unk.txt'
bpali_scp_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.bpali.scp'
# bpali_scp_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-dev.bpali.scp'
bpali_path = '/home/htang2/proj/data/wsj/ext-data/train-si284.bpali'   
norm_path = '/home/htang2/proj/data/wsj/ext-data/si284-0.9-train.mean-var'


with open(phone_unk_txt_path, 'r') as phone_file:
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

class WSJDataset(Dataset):
    def __init__(self, data_path,bpali_path, transform=None, target_transform=None):
        super().__init__()
        self.data_path = data_path
        self.bpali_path = bpali_path
        self.fbank = []
        self.transform = transform
        self.target_transform = target_transform
                
        bpali_txt = open(bpali_path, 'r')
        data_file=open(self.data_path,'r') 
        lines = data_file.readlines()
        count = 0
        for line in lines:
            count += 1
            file_index = line.split(' ')[0].strip() # 011c0201
            line = line.split(' ')[1].strip()

            fbank_path = line.split(':')[0]
            fbank_index = line.split(':')[1]
            
            # utter
            f = open(fbank_path, 'rb')
            f.seek(int(fbank_index))
            mat = kaldiark.parse_feat_matrix(f)
            mat = (mat - mean) / std
#             mat = np.expand_dims(mat, axis=0)
            mat = torch.Tensor(mat).to(device)
            
        
            # labels
            bpali_txt.seek(int(bp_dict[file_index]))
            bpali_txt_lines = bpali_txt.readline().strip()
            phones = bpali_txt_lines.split(' ')
            labels = [phone2id[p] for p in phones]
            labels = torch.LongTensor(labels).to(device)        
            
            self.fbank.append((file_index, mat, labels))
            
            
    def __getitem__(self,index):
        file_index, mat, labels = self.fbank[index]
        return file_index, mat, labels
    
    def __len__(self): 
        return len(self.fbank)
    