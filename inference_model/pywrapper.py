import torch
from torch import nn
import torch.nn.functional as F
from typing import MutableSequence
import numpy as np
import pandas as pd
import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'softplus':
        return nn.Softplus()
    elif activation == 'identity':
        return nn.Identity()
    else:
        raise NotImplementedError

def get_normalization(normalization, normalized_shape):
    if normalization == 'ln':
        return nn.LayerNorm(normalized_shape)
    elif normalization == 'bn1d':
        return nn.BatchNorm1d(normalized_shape)
    elif normalization == 'bn2d':
        return nn.BatchNorm2d(normalized_shape)
    else:
        raise NotImplementedError

class MLP(nn.Module):
    def __init__(self, 
            in_size, 
            hidden_sizes, 
            dropout=0,
            normalization=False,
            activation='relu', 
            last_dropout=None,
            last_normalization=None,
            last_activation=None):
        super(MLP, self).__init__()
        
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        if last_dropout is None:
            last_dropout = dropout
        if last_normalization is None:
            last_normalization = normalization
        if last_activation is None:
            last_activation = activation
        
        assert isinstance(hidden_sizes, MutableSequence)     
        
        layers = []
        num_layers = len(hidden_sizes)

        for l, hid_size in enumerate(hidden_sizes):
            # add linear layer
            layers.append(nn.Linear(in_size, hid_size))

            # add normalization
            norm = normalization if l < num_layers - 1 else last_normalization
            if norm:
                layers.append(get_normalization(norm, hid_size))

            # add activation
            act = activation if l < num_layers - 1 else last_activation
            layers.append(get_activation(act))

            # add dropout
            p = dropout if l < num_layers - 1 else last_dropout
            if p > 0:
                layers.append(nn.Dropout(p))

            in_size = hid_size

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)



class LFP2INT(nn.Module):
    def __init__(self, meta, optimizer, loss):
        super(LFP2INT, self).__init__()
        self.emb = MLP(
                    in_size = meta['input_dims'],
                    hidden_sizes=512,
                    activation='relu'
                    )
        self.core = MLP(
                    in_size = 512,
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
        y_pred = F.softmax(y_pred, dim=-1)
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

model = LFP2INT({'input_dims':961, 'output_dims':3, 'lr':0.0001, 'weight_decay': 0.00001}, torch.optim.Adam, torch.nn.CrossEntropyLoss())
#model.load_state_dict(torch.load('int2lfp_model.pth'))
train_mu = np.load('train_mu.npy')
print(train_mu.shape)


# assume inp is 1 x Time x Electrodes
def forwardpass(inp):
    global model, train_mu
    inp = np.random.rand(1,6000,31)
    print(inp.shape)
    inp = np.dstack([inp[...,:19], inp[...,20:]])
    print(inp.shape)
    inp = inp - train_mu
    mat = []
    for k in range(inp.shape[-1]):
        X = smooth(inp[0, :, k], 250)
        mat.append(X)
    mat = np.stack(mat)
    mymat = pd.DataFrame(mat.transpose()).corr(method='pearson').to_numpy()

    model_input = torch.tensor(mymat.flatten(), dtype=torch.float32)
    pred, _ = model(model_input)
    pred = torch.argmax(pred, dim=-1).detach().cpu().numpy()
    
    return pred
