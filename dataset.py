import pickle
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MalariaDataset(Dataset):
    def __init__(self):
        datafile = open('./data.pickle', 'rb')
        self.image_index = pickle.load(datafile)
        self.labels = pickle.load(datafile)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = Image.open(self.image_index[index])
        arr = np.array(img) / 255
        label = self.labels[index]
        return {'image': arr, 'label': label}
