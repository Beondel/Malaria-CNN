from torch import nn
from torchvision import transforms
from dataset import MalariaDataset
from classifier import CNN
from torch.utils.data import DataLoader
from tqdm import trange
from torch import optim
import matplotlib.pyplot as plt

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
    model = CNN()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    loss_history = []
    for epoch in range(1):
        for i, data in enumerate(loader):
            X, Y = data
            optimizer.zero_grad()
            Y_hat = model(X)
            loss = criterion(Y_hat.float(), Y.float())
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'loss: {loss.item()}')
                loss_history.append(loss.item())

    plt.plot(range(len(loss_history)), loss_history)
    plt.show()