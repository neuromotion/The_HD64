import torch, os
from torch import nn
from config import Config
import argparse
from datamodules.stim2emg import StimToEMGDataModule, read_dataset
from datamodules.stim2sense import StimToSenseDataModule
from datamodules.int2lfp import IntToLFPDataModule

from models import Stim2EMG
from trainer import Trainer

from joblib import Parallel, delayed
import itertools, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# what do we want?
# show target EMG (maybe raw, maybe discretized)
# show what the network thinks it could achieve
# show what was actually achieved


'''Demo day retest
python analyse_stim2emg_retest.py --exp_type StimToEMG --dataset LiveDemo --exp_name DARPA
PROP_JSON = '/media/data_cifs/projects/prj_deepspine/isi-c/live_demo_proposals.json'
PARQ_FILE = '/media/data_cifs_lrs/projects/prj_deepspine/isi-motioncap/202109281300-JG/EMG_Retest_Set1.parquet'
EMG_Retest_Set2.parquet
'''

PROP_JSON = '/media/data_cifs/projects/prj_deepspine/isi-c/live_demo_proposals.json' #proposals/collated_proposals_day3.json'
PARQ_FILE = '/media/data_cifs/projects/prj_deepspine/hd64/ckpts/StimToEMG/HD64RetestSession1/RetestData.parquet'
N_PROPS = 10
N_TARGETS = 16
N_REPEATS = 5

def run(cfg):
    # fix random seed
    torch.manual_seed(cfg.seed)
    # prepare the data and model
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
    dm.prepare_data(exp_name = cfg.dataset)
    dm.setup(exp_name = cfg.dataset)

    optimizer = torch.optim.Adam
    loss = torch.nn.L1Loss()

    model = Stim2EMG(
                meta = cfg.stim2emg,
                optimizer = optimizer,
                loss = loss)

    trainer = Trainer(cfg)

    df = pd.read_parquet(PARQ_FILE, engine='pyarrow')
    
    stim_start = 't_0_049846'
    stim_end = 't0_50002' 

    emgTF = dm.transformers['emg']['discretization']

    EMG_LABS = ['L-PL', 'L-GA', 'L-BF', 'R-PL', 'R-GA', 'R-BF']

    gt_mask = (df['ProposalID'] == 'gt_ees')

    all_coefs = []

    ood_elecs = [0, 3, 12, 43, 47, 54, 133, 134, 137, 140, 169, 171, 177, 178, 179, 181, 182, 186, 187, 188]

    seen_errors, unseen_errors = [], []

    # for each target
    for target_id in range(N_TARGETS):
        target_mask = (df['TargetIdx'] == int(target_id+1)) #'proposal_{}'.format(prop_id))
        #trial_ids = np.unique(df.loc[p_mask]['TrialNum'])

        # get the ground truth re-stim
        GT_EMG = []
        gt_trials = np.unique(df.loc[gt_mask & target_mask]['TrialNum'])
        for trial in gt_trials:
            t_mask = (df['TrialNum'] == trial)

            emg = []
            for m in cfg.selected_muscles:
                m_mask = (df['ChannelID'] == int(m)) 
                samples = df.loc[t_mask & m_mask].iloc[:, df.columns.get_loc(stim_start):df.columns.get_loc(stim_end)].to_numpy()
                mean = samples.mean(1, keepdims=True)
                emg.append(samples - mean)

            emg = np.stack(emg, axis=2)
            tf_emg = emgTF.transform(emg)
            GT_EMG.append(tf_emg)

        GT_EMG = np.vstack(GT_EMG).mean(0, keepdims=True)
        
        # get the retest EMGs
        all_emgs = []
        seen_elec, unseen_elec = [], []

        for prop_id in range(N_PROPS):
            p_mask = (df['ProposalID'] == 'proposal_{}'.format(prop_id))
            trial_ids = np.unique(df.loc[p_mask & target_mask]['TrialNum'])

            # for each inferred stimulation repeat        
            for trial in trial_ids:
                t_mask = (df['TrialNum'] == trial)
                emg = []

                for m in cfg.selected_muscles:
                    m_mask = (df['ChannelID'] == int(m)) 
                    samples = df.loc[t_mask & m_mask].iloc[:, df.columns.get_loc(stim_start):df.columns.get_loc(stim_end)].to_numpy()
                    mean = samples.mean(1, keepdims=True)
                    emg.append(samples - mean)

                emg = np.stack(emg, axis=2)
                tf_emg = emgTF.transform(emg)
                all_emgs.append(tf_emg)

                if np.unique(df['StimElec'][t_mask])[0] in ood_elecs:
                    unseen_elec.append(tf_emg)
                else:
                    seen_elec.append(tf_emg)

        ### COMPUTE PER CHANNEL ERROR
        for emg in seen_elec:
            seen_errors.append(np.abs(emg.squeeze() - GT_EMG.squeeze()))
        for emg in unseen_elec:
            unseen_errors.append(np.abs(emg.squeeze() - GT_EMG.squeeze()))


        EMGS = np.vstack(all_emgs)

        coefs = [np.corrcoef(EMGS[k,:], GT_EMG.squeeze())[0,1] for k in range(EMGS.shape[0])]
        all_coefs.extend(coefs)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        X = np.zeros((5 + EMGS.shape[0],6)) + 0.5
        X[0, :] = GT_EMG
        X[5:, :] = EMGS

        ax.imshow(X, cmap=plt.get_cmap('seismic'), vmin=0., vmax=1.)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(EMG_LABS, rotation=45, fontsize=8)

        #ax.set_yticks(np.arange(X.shape[0]+1)-.5, minor=True)
        ax.set_yticks([])

        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        plt.tight_layout()
        #plt.show()
        plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(all_coefs, bins=20)

    ax.set_xlabel(r'Pearson $r$', fontweight='bold', fontsize=14)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    plt.show()


    import scipy.io as S
    X = {'seen_errors': np.stack(seen_errors), 'unseen_errors': np.stack(unseen_errors)}
    S.savemat('hd64retest1_analysis_errors.mat', X)

    '''
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.boxplot(np.stack(seen_errors))
    ax = fig.add_subplot(122)
    ax.boxplot(np.stack(unseen_errors))
    plt.show()
    '''

    print('hello')

if __name__ == "__main__":
    # instantiate the configuration class
    cfg = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', required=True, help='indicates the data i-o mapping')
    parser.add_argument('--dataset', required=True, help='name of the dataset')
    parser.add_argument('--exp_name', required=True, help='prefix for this particular run')

    args = parser.parse_args().__dict__
    for key in args.keys():
        setattr(cfg, key, args[key])

    run(cfg)
