import pickle
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class MalariaDataset(Dataset):
    def __init__(self, transforms=None):
        datafile = open('./data.pickle', 'rb')
        self.image_index = pickle.load(datafile)
        self.labels = pickle.load(datafile)
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = Image.open(self.image_index[index])
        label = self.labels[index]
        if self.transforms:
            img = self.transforms(img)
        return img, label
    
