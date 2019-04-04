import torch
from torchvision import transforms
from dataset import MalariaDataset
from classifier import CNN
from torch.utils.data import DataLoader
from tqdm import trange

if __name__ == '__main__':
    transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.25, 0.25, 0.25])
    ])

    dataset = MalariaDataset(transforms=transforms)

    loader = DataLoader(dataset, batch_size=64,
                        shuffle=True, num_workers=4)
    
