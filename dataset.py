from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os 
from PIL import Image
from torchvision import transforms
import torch 
import numpy as np

class DamformerDataset(Dataset):
    def __init__(self,datapath, opt):
        self.datapath = datapath
        self.files = os.listdir(os.path.join(self.datapath,'sar'))
        self.sar_path = os.path.join(self.datapath,'sar')
        self.opt_path = os.path.join(self.datapath, opt)
        self.label_path = os.path.join(self.datapath,'label')
        

    def __len__(self):
        return len(self.files)
    
    def readimg(self,path):
        img = Image.open(path).convert("RGB")
        tensor = transforms.ToTensor()(img)
        return tensor
    
    def read_label(self, path):
        label = Image.open(path).convert("L")   # 单通道
        label = torch.from_numpy(np.array(label)).long()
        return label
        
    def __getitem__(self, idx):
        sar = self.readimg(os.path.join(self.sar_path,self.files[idx]))
        opt = self.readimg(os.path.join(self.opt_path,self.files[idx]))
        label = self.read_label(os.path.join(self.label_path,self.files[idx]))
        return sar,opt,label,self.files[idx]
   
# if __name__ == '__main__': 
#     path = 'D:/zjn/code/DamFormer/train'
#     dataset = DamformerDataset(path)
#     traindataloder = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

#     for sar,opt,label,filename in traindataloder:
#         print(sar.shape,opt.shape,label.shape,filename)
#         break