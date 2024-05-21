import torch
from torch import nn
import torch.nn.functional as F
from layers.feedforward import MLP

class Stim2EMG(nn.Module):
    def __init__(self, meta, optimizer, loss):
        super(Stim2EMG, self).__init__()
        self.emb = MLP(
                    in_size = meta['input_dims'],
                    hidden_sizes=16,
                    activation='relu'
                    )
        self.core = MLP(
                    in_size = 16,
                    hidden_sizes = 256,
                    dropout = 0.5,
                    activation = 'relu'
                    )
        self.readout = MLP(
                    in_size = 256,
                    hidden_sizes = meta['output_dims'],
                    activation = 'identity'
                    )
        
        self.optimizer = optimizer(self.parameters(), lr=meta['lr'], weight_decay=meta['weight_decay'])
        self.criterion = loss

    def forward(self, x):
        h = self.emb(x)
        h = self.core(h)
        y_pred = self.readout(h)

        return y_pred, h

    def _update(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        x, y = batch
        y_pred, _ = self(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _inference(self, batch):
        self.eval()

        with torch.no_grad():
            x, y = batch
            y_pred, z = self(x)

            return {'x': x, 'y_pred': y_pred, 'y': y, 'z': z}


class Net(nn.Module):
    def __init__(self, optimizer, loss):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, [30, 30])
        self.bn = torch.nn.BatchNorm2d(4)

        self.pool = nn.MaxPool2d((20, 1))
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(4, 1, [5, 1])
        self.fc1 = nn.Linear(48, 32) #48 #1178
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)
        self.softmax = nn.Softmax(dim=-1)

        self.optimizer = optimizer(self.parameters(), lr=1e-3, weight_decay=1e-5)
        self.criterion = loss

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)

        return x

    def _update(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()
