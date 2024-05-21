import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
import itertools
import os
import math
import json
import hashlib
import matplotlib.pyplot as plt
from utils.visualization import heatmap, annotate_heatmap
import tqdm

def read_dataset(data_dir, selected_muscles, ignore_electrodes):
    df = pd.read_parquet(data_dir, engine='pyarrow')
#     stim_start, stim_end = 't_0_049846', 't0_50002' 
    stim_start, stim_end = 't0', 't0_5' ##### 231024

    electrodes = np.unique(df['StimElec1'])
    channels = np.unique(df['ChannelID'])
    amplitudes = np.unique(df['StimAmp'])
    frequencies = np.unique(df['StimFreq'])

    # allocate lists to hold data from each trial
    configs = []
    for e in tqdm.tqdm(electrodes):
        if e in ignore_electrodes:
            continue

        e_mask = (df['StimElec1'] == e)
        for a in amplitudes:
            
            a_mask = (df['StimAmp'] == a)
            for f in frequencies:
                f_mask = (df['StimFreq'] == f)
                
                emg = []
                for i, m in enumerate(selected_muscles):
                    m_mask = (df['ChannelID'] == int(m))
                    samples = df.loc[e_mask & m_mask & a_mask & f_mask]
                    
                    ##### 231024
                    samples = samples.iloc[:, df.columns.get_loc(stim_start):df.columns.get_loc(stim_end)].to_numpy()
                    if samples.shape[0] != 4:
                        samples = samples[-4:, :]                                        
                    mean = samples.mean(1,keepdims=True)                       
                    if samples.shape[0] != 4:
                        print(e, a, f, m, samples.shape[0])

                    # mean subtraction
                    emg.append(samples - mean)
                
                emg = np.stack(emg, axis=2) # repetition x time x muscle
                if emg.shape[0] > 0:
                    config = {
                        'ees': {
                            'freq': f,
                            'amp': a,
                            'elec': {
                                'idx': int(e),
                                'x': np.unique(df[df['StimElec1'] == e]['StimXCoord1'])[0],
                                'y': np.unique(df[df['StimElec1'] == e]['StimYCoord1'])[0]
                            },
                        },
                        'emg': emg.astype(np.float32)
                    }
                    configs.append(config)

    return configs


def cov_corr(x, y, axis):
    centered_x = x - x.mean(axis, keepdims=True)
    centered_y = y - y.mean(axis, keepdims=True)

    output_std = np.sqrt((centered_x ** 2).mean(axis))
    target_std = np.sqrt((centered_y ** 2).mean(axis))
    
    cov = (centered_x * centered_y).mean(axis)
    corr = cov / (output_std * target_std)
    
    return cov, corr


def split_half_reliability(
        configs,
        axis,
        num_iterations=100,
    ):

    train_emg = []
    valid_emg = []

    for conf in configs:
        emg = conf['emg']
        num_trials = emg.shape[0]
        for i in range(num_iterations):
            shuffled_indices = np.random.permutation(np.arange(num_trials))
            split = int(np.round(shuffled_indices.shape[0] * 0.5))

            train_indices = shuffled_indices[:split]
            valid_indices = shuffled_indices[split:]

            # averaged over repetitions
            train_emg.append(np.take(emg, train_indices, axis=0).mean(0))
            valid_emg.append(np.take(emg, valid_indices, axis=0).mean(0))

    train_emg = np.stack(train_emg, axis=0)
    valid_emg = np.stack(valid_emg, axis=0)
    train_emg = train_emg.reshape(len(configs), num_iterations, *train_emg.shape[1:])
    valid_emg = valid_emg.reshape(len(configs), num_iterations, *valid_emg.shape[1:])

    cov, corr = cov_corr(train_emg, valid_emg, axis=(axis+1))
    
    # average over iterations
    cov = cov.mean(1)
    corr = corr.mean(1)
    
    stats = {
        'cov': cov,
        'corr': corr
    }

    return stats


def filtering(
        configs,
        cov,        # configs x channel
        corr,       # configs x channel
        cov_threshold,
        corr_threshold, 
    ):

    # scaling covariance
    transformer = preprocessing.RobustScaler(quantile_range=(5, 95)).fit(cov)
    cov = (transformer.transform(cov))

    corr_mask = corr > corr_threshold     # filter unreliable EMG
    cov_mask = cov > cov_threshold        # filter subthreshold EMG
    # select configuration if at least one of the channels is above threshold
    mask = np.any(corr_mask * cov_mask, axis=1)

    print('noise ceiling before filtering: %.4f' % corr.mean())
    print('noise ceiling after filtering: %.4f' % corr[mask].mean())
    
    selected_configs = [conf for conf, m in zip(configs, mask) if m]

    return selected_configs


class Discretization():
    def __init__(self, q=99):
        self.q = q
        self.percentile = None

    def fit(self, x):
        # x: trial x time x channel
        rectified_x = np.abs(x)
        discretized_x = rectified_x.mean(1)     # trial x channel

        self.percentile = np.stack([np.percentile(discretized_x[:,i], self.q) for i in range(discretized_x.shape[-1])])

    def transform(self, x):
        assert self.percentile is not None

        print('Raw: ', x.shape)
        # x: trial x time x channel
        rectified_x = np.abs(x)
        discretized_x = rectified_x.mean(1)     # trial x channel
        discretized_x = discretized_x / self.percentile[None,:]

        # clamps all elements in discretized_x to be smaller or equal 1
        # so that the range of discretized_x becomes [0, 1]
        discretized_x[discretized_x>1] = 1
        print('Discrit: ', discretized_x.shape)
        
        return discretized_x
        

class Parameterization():
    def __init__(self, elec_encoding):
        assert elec_encoding in ['onehot', 'pos']
        self.elec_encoding = elec_encoding

    def fit(self, x):
        # scaling freq and amp to [0-1]
        self.freq_scale = 1. / x['freq'].max()
        self.amp_scale = 1. / x['amp'].max()

        # scaling freq and amp to [0.-0.9]
        self.freq_scale = self.freq_scale * 0.9
        self.amp_scale = self.amp_scale * 0.9
        self.freq_bias = 0.
        self.amp_bias = 0.

        if self.elec_encoding == 'onehot':
            self.elec2idx = {}
            self.idx2elec = {}
            electrodes = np.unique(x['elec']['idx'])
            self.num_electrodes = len(electrodes)
            for idx, e in enumerate(electrodes):
                # one_hot = np.zeros(self.num_electrodes)
                # one_hot[idx] = 1
                self.elec2idx[e] = 2**idx
                self.idx2elec[2**idx] = e

    def transform(self, x):
        assert self.freq_scale is not None
        assert self.amp_scale is not None

        freq = x['freq'][:,None]
        amp = x['amp'][:,None]

        if self.elec_encoding == 'onehot':
            # convert integer to binary vector
            elec = np.vectorize(self.elec2idx.get)(x['elec']['idx'])
            elec = ((elec[:,None] & (1 << np.arange(self.num_electrodes))) > 0).astype(int)
        else:
            #elec = x['elec']['pos']
            elec = np.vstack([x['elec']['x'], x['elec']['y']]).T

        parameterized_x = np.concatenate([freq, amp, elec], axis=1)
        parameterized_x[:,0] = (parameterized_x[:,0] * self.freq_scale) + self.freq_bias
        parameterized_x[:,1] = (parameterized_x[:,1] * self.amp_scale) + self.amp_bias

        return parameterized_x

    def inverse_transform(self, parameterized_x):
        freq = (parameterized_x[:,0] - self.freq_bias) / self.freq_scale
        amp = (parameterized_x[:,1] - self.amp_bias) / self.amp_scale

        if parameterized_x.shape[1] == 2:
            return {
                'freq': freq,
                'amp': amp
            }

        if self.elec_encoding == 'onehot':
            assert parameterized_x[:,2:].shape[1] == self.num_electrodes

            # convert binary vector to integer
            elec = (parameterized_x[:,2:] * (1 << np.arange(self.num_electrodes))).sum(1).astype(int)
            elec = np.vectorize(self.idx2elec.get)(elec)

            return {
                'freq': freq,
                'amp': amp,
                'elec': elec
            }
        else:
            return {
                'freq': freq,
                'amp': amp,
                'elec': parameterized_x[:,2:]
            }

def preprocess(train_data, test_data, preprocess_cfg):

    # preprocess EMG
    ## dicretization
    discretization = Discretization(**preprocess_cfg['emg'])
    discretization.fit(train_data['emg'])
    train_data['transformed_emg'] = discretization.transform(train_data['emg'])
    test_data['transformed_emg'] = discretization.transform(test_data['emg'])

    # preprocess EES
    ## parameterization
    # num_electrodes = train_data['ees'].shape[-1]
    parameterization = Parameterization(**preprocess_cfg['ees'])
    parameterization.fit(train_data['ees'])
    
    train_data['transformed_ees'] = parameterization.transform(train_data['ees'])
    test_data['transformed_ees'] = parameterization.transform(test_data['ees'])
    
    return {
        'emg': {
            'discretization': discretization
        },
        'ees': {
            'parameterization': parameterization,
        }
    }


def train_test_split(configs, holdout_frequencies, holdout_amplitudes):
    train_configs = []
    test_configs = []

    for conf in configs:
        if conf['ees']['freq'] in holdout_frequencies or conf['ees']['amp'] in holdout_amplitudes:
            test_configs.append(conf)
        else:
            train_configs.append(conf)
        
    train_data = {
        'ees': {
            'freq': np.array([conf['ees']['freq'] for conf in train_configs], dtype=np.float32),
            'amp': np.array([conf['ees']['amp'] for conf in train_configs], dtype=np.float32),
            'elec': {
                'idx': np.array([conf['ees']['elec']['idx'] for conf in train_configs], dtype=np.int32),
            'x': np.array([conf['ees']['elec']['x'] for conf in train_configs], dtype=np.float32),
            'y': np.array([conf['ees']['elec']['y'] for conf in train_configs], dtype=np.float32)
            },
        },
        'emg': np.concatenate([conf['emg'].mean(0, keepdims=True) for conf in train_configs], axis=0),
        'config': np.array([idx for idx, conf in enumerate(train_configs)], dtype=np.int32)
    }

    test_data = {
        'ees': {
            'freq': np.array([conf['ees']['freq'] for conf in test_configs], dtype=np.float32),
            'amp': np.array([conf['ees']['amp'] for conf in test_configs], dtype=np.float32),
            'elec': {
                'idx': np.array([conf['ees']['elec']['idx'] for conf in test_configs], dtype=np.int32),
                'x': np.array([conf['ees']['elec']['x'] for conf in test_configs], dtype=np.float32),
                'y': np.array([conf['ees']['elec']['y'] for conf in test_configs], dtype=np.float32)
            },
        },
        'emg': np.concatenate([conf['emg'].mean(0, keepdims=True) for conf in test_configs], axis=0),
        'config': np.array([idx for idx, conf in enumerate(test_configs)], dtype=np.int32)
    }
    
    print("Train: ", np.unique(train_data['ees']['freq']))
    print("Test: ", np.unique(test_data['ees']['freq']))
    
    return train_data, test_data

class StimToEMGDataModule():
    def __init__(self, 
        data_dir,
        selected_muscles,
        ignore_electrodes,
        elec_encoding,
        q,
        holdout_frequencies,
        holdout_amplitudes,
        train_bs,
        valid_bs,
        test_bs,
        seed=1992,
        device='cpu'
    ):
        self.data_dir = data_dir.rstrip('//')
        self.selected_muscles = selected_muscles
        self.ignore_electrodes = ignore_electrodes
        self.elec_encoding = elec_encoding
        self.q = q
        self.holdout_frequencies = holdout_frequencies
        self.holdout_amplitudes = holdout_amplitudes
        self.train_bs = train_bs
        self.valid_bs = valid_bs
        self.test_bs = test_bs
        self.seed = seed
        self.device = device
        
        # fix random seed
        np.random.seed(self.seed)

    def prepare_data(self, exp_name, raw_data): ##### 231024
        _dir = os.path.join(self.data_dir, 'StimToEMG', exp_name)
        _rdir = os.path.join(self.data_dir, exp_name)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        save_path = os.path.join(_dir, 'data.npy')

        if False: #os.path.isfile(save_path):
            print("Here it is: ", save_path)
            data = np.load(save_path, allow_pickle=True).item()

            '''
            import ipdb; ipdb.set_trace()
            # compute reliability
            configs = read_dataset(
                _rdir + 'StimToEMG.parquet', 
                selected_muscles=self.selected_muscles,
                ignore_electrodes=self.ignore_electrodes
            )
            '''
        else:
            configs = read_dataset(
#                 _rdir + 'StimToEMG.parquet', 
                raw_data, ##### 231024
                selected_muscles=self.selected_muscles,
                ignore_electrodes=self.ignore_electrodes
            )

            train_data, test_data = train_test_split(
                configs,
                holdout_frequencies=self.holdout_frequencies,
                holdout_amplitudes=self.holdout_amplitudes
            )

            transformers = preprocess(
                train_data, 
                test_data, 
                preprocess_cfg={
                    'ees': {
                        'elec_encoding': self.elec_encoding
                    },
                    'emg': {
                        'q': self.q
                    }
            })

            # save test EMG figure
            def visualize_dataset(file, data):
                transformed_emg =[]
                legend = []

                for c in np.unique(data['config']):
                    freq = data['ees']['freq'][data['config'] == c][0]
                    amp = data['ees']['amp'][data['config'] == c][0]
                    elec = data['ees']['elec']['idx'][data['config'] == c][0]
                    transformed_emg.append(data['transformed_emg'][data['config'] == c].mean(0))
                    
                    if transformers['ees']['parameterization'].elec_encoding == 'onehot':
                        idx = transformers['ees']['parameterization'].elec2idx[elec]
                        legend.append('[%d]: %.0fHz, %.0fuA, %d(elec)/%d(idx)' % (c, freq, amp, elec, int(math.log(idx, 2))))
                    else:
                        x, y = data['ees']['elec']['x'][data['config'] == c][0], data['ees']['elec']['y'][data['config'] == c][0]
                        legend.append('[%d]: %.0fHz, %.0fuA, %d(elec)/[%.1f, %.1f](pos)' % (c, freq, amp, elec, x, y))
                
                transformed_emg = np.stack(transformed_emg, axis=0)

#                 fig, ax = plt.subplots(figsize=(transformed_emg.shape[1],transformed_emg.shape[0]))
                fig_temp, ax_temp = plt.subplots(figsize=(100,100))
                im = ax_temp.imshow(transformed_emg)
        
                ax_temp.set_xticks(np.arange(transformed_emg.shape[1]))
                ax_temp.set_yticks(np.arange(transformed_emg.shape[0]))
                # ... and label them with the respective list entries.
                ax_temp.set_xticklabels(self.selected_muscles)
                ax_temp.set_yticklabels(legend)
        
#                 fig, ax = plt.subplots(figsize=(5,5))
#                 im, cbar = heatmap(transformed_emg, legend, self.selected_muscles, ax=ax, vmin=0, vmax=0.9, cmap="YlGn", cbarlabel="muscle recruitment")
#                 ax.set_xticklabels([])
#                 ax.set_yticklabels([])
#                 texts = annotate_heatmap(im, valfmt="{x:.2f}")
#                 cbar.remove()
#                 fig.tight_layout()
                plt.savefig(file)
                print("Save figure: ", file)
#                 plt.show() ##### 231024
            
            visualize_dataset(os.path.join(_dir, 'train_dataset.png'), train_data)
            visualize_dataset(os.path.join(_dir, 'test_dataset.png'), test_data)
            
            print('save_path: ' + save_path)
            np.save(save_path, {
                'train_data': train_data,
                'test_data': test_data,
                'transformers': transformers
            })

    def setup(self, exp_name, side='both'):
        assert side in ['left', 'right', 'both']

        _dir = os.path.join(self.data_dir, 'StimToEMG', exp_name)
        data = np.load(os.path.join(_dir, 'data.npy'), allow_pickle=True).item()
        print("Dir: " + os.path.join(_dir, 'data.npy'))
        train_data = data['train_data']
        test_data = data['test_data']

        # select left or right side EMG channels
        if side in ['left', 'right']:
            s = 'L' if side == 'left' else 'R'
            train_data['transformed_emg'] = train_data['transformed_emg'][:,[ch[0]==s for ch in self.selected_muscles]]
            test_data['transformed_emg'] = test_data['transformed_emg'][...,[ch[0]==s for ch in self.selected_muscles]]
        
        # numpy array to torch tensor
        train_ees = torch.Tensor(train_data['transformed_ees'])
        train_emg = torch.Tensor(train_data['transformed_emg'])
        test_ees = torch.Tensor(test_data['transformed_ees'])
        test_emg = torch.Tensor(test_data['transformed_emg'])

        # device conversion
        train_ees  = train_ees.to(self.device)
        train_emg  = train_emg.to(self.device)
        test_ees  = test_ees.to(self.device)
        test_emg  = test_emg.to(self.device)

        self.train_dataset = TensorDataset(train_ees, train_emg)
        self.valid_dataset = TensorDataset(test_ees, test_emg)
        self.test_dataset = TensorDataset(test_ees, test_emg)

        # transformers used for preprocessing EMG and EES
        self.transformers = data['transformers']
        
        print(self.transformers['emg']['discretization'])
        
    def inverse_transform_ees(self, X):
        x = X.copy()

        parameterization = self.transformers['ees']['parameterization']
        x = parameterization.inverse_transform(x)
        
        return x

    def train_dataloader(self):
        train_bs = len(self.train_dataset) if self.train_bs is None else self.train_bs
        return DataLoader(self.train_dataset, batch_size=train_bs, shuffle=True, num_workers=0, drop_last=True)

    def valid_dataloader(self):
        valid_bs = len(self.valid_dataset) if self.valid_bs is None else self.valid_bs
        return DataLoader(self.valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=0, drop_last=False)

    def test_dataloader(self):
        test_bs = len(self.test_dataset) if self.test_bs is None else self.test_bs
        return DataLoader(self.test_dataset, batch_size=test_bs, shuffle=False, num_workers=0, drop_last=False)

