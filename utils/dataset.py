# coding=utf-8
# @Author  : Mohammadreza Qaraei
# @Email   : mohammadreza.mohammadniaqaraei@aalto.fi


from numpy import float32
from torch.utils.data import Dataset

class RnDataset(Dataset):
    def __init__(self, data, labels):
        super(RnDataset, self).__init__()
        self.data = data
        self.labels = labels
    
    def __getitem__(self, index):
        data = self.data[index].astype(float32)
        labels = self.labels[index]
        return data, labels

    def __len__(self):
        return self.data.shape[0]
