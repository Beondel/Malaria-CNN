import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 16 * 16)
        self.fc2 = nn.Linear(16 * 16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, X):
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = X.view(-1, 16 * 16 * 16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.softmax(self.fc3(X), dim=0)
        return X