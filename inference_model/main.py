import torch, os
from torch import nn
from config import Config
import argparse
from datamodules.stim2emg import StimToEMGDataModule

from models import Stim2EMG
from trainer import Trainer

from joblib import Parallel, delayed
import itertools

def run(cfg):
    # fix random seed
    torch.manual_seed(cfg.seed)
    # prepare the data and model
    if cfg.exp_type == 'StimToEMG':
        dm = StimToEMGDataModule(
                    data_dir = cfg.data_dir,
                    selected_muscles = cfg.selected_muscles,
                    ignore_electrodes = cfg.ignore_electrodes,
                    elec_encoding = cfg.elec_encoding,
                    q = 99.,
                    holdout_frequencies = cfg.holdout_frequencies,
                    holdout_amplitudes = cfg.holdout_amplitudes,
                    train_bs = cfg.train_bs,
                    valid_bs = cfg.valid_bs,
                    test_bs = cfg.test_bs
             )
        dm.prepare_data(exp_name = cfg.dataset, raw_data = cfg.raw_data)
        dm.setup(exp_name = cfg.dataset)

        optimizer = torch.optim.Adam
        loss = torch.nn.L1Loss()

        model = Stim2EMG(
                    meta = cfg.stim2emg,
                    optimizer = optimizer,
                    loss = loss)

        trainer = Trainer(cfg)

    else:
        raise NotImplementedError

    if cfg.mode == 'train':
        trainer.train(model, datamodule=dm)
    elif cfg.mode == 'eval':
        trainer.eval(model, datamodule=dm)
    elif cfg.mode == 'inference':
        '''
        all_pairs = itertools.product(*[cfg.electrode_index, cfg.target_index])
        Parallel(n_jobs=8)(delayed(trainer.inference)(model, datamodule=dm, electrode_index=pair[0], target_index=pair[1]) for pair in all_pairs)
        '''
        
        trainer.inference(
            model, 
            datamodule=dm,
            electrode_index=cfg.electrode_index,
            target_index=cfg.target_index 
        )
    else:
        raise NotImplementedError
            
'''
Usage:
python main.py --exp_type StimToEMG --dataset demo --exp_name demoS2E --mode train
'''
if __name__ == "__main__":
    # instantiate the configuration class
    cfg = Config()

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_type', required=True, help='indicates the data i-o mapping')
    parser.add_argument('--dataset', required=True, help='name of the dataset')
    parser.add_argument('--exp_name', required=True, help='prefix for this particular run')
    parser.add_argument('--mode', required=True, help='train/eval/inference')

    parser.add_argument('--raw_data', required=True, help='directory of raw data') ##### 231024
    
    # these are only relevant for the inference stage
    parser.add_argument('--electrode_index', type=int, nargs='+', default=[0])
    parser.add_argument('--target_index', type=int, nargs='+', default=[0])

    args = parser.parse_args().__dict__
    for key in args.keys():
        setattr(cfg, key, args[key])

    run(cfg)
