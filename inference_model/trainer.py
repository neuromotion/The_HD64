import torch, os, logging, json, time, pickle
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from utils.visualization import tsne
from param_recovery import GlobalSimulator, LocalSimulator, Inference
import matplotlib.pyplot as plt
from scipy import stats

class TimeoutException(Exception):   ##### 231024 # Custom exception class 
    pass
def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

class Trainer():
    def __init__(self, cfg):
        self.num_epochs = cfg.num_epochs
        self.eval_interval = cfg.eval_interval
        self.device = cfg.device
        self.deterministic = cfg.deterministic
        self.checkpoint_dir = cfg.ckpt_dir
        self.exp_type = cfg.exp_type
        self.exp_name = cfg.exp_name

        # inference parameters are needed only for retest experiments
        if cfg.exp_type in ['StimToEMG', 'StimToSensation']: 
            self.num_rounds = cfg.num_rounds 
            self.num_simulations = cfg.num_simulations
            self.simulation_batch_size = cfg.simulation_batch_size
            self.training_batch_size = cfg.training_batch_size
            self.num_samples = cfg.num_samples
            self.filtering_ratio = cfg.filtering_ratio
            self.num_proposals = cfg.num_proposals
            self.timeout = cfg.timeout

    def train(self, model, datamodule):
        _dir = os.path.join(self.checkpoint_dir, self.exp_type)
        if not os.path.exists(_dir):
            os.makedirs(_dir)

        model = model.to(self.device)

        losses = []
        train_dataloader = datamodule.train_dataloader()
        valid_dataloader = datamodule.valid_dataloader()
        print("Before training..")
        for epoch in range(self.num_epochs):
            for batch, data in enumerate(train_dataloader):
                x, y = data

                # add a jitter here to account for the area of the stimulation contact
                _r = torch.normal(0, 0.01, size=(x.shape[0], 2))
                x[:,2:] = x[:,2:] + _r

                x = x.to(self.device)
                y = y.to(self.device)

                lossval = model._update((x,y))
                log = 'Batch: %d | Epoch: %d | loss= %f' % (batch, epoch, lossval)
                losses.append(lossval)
                print(log)

        # save the model 
        torch.save(datamodule, os.path.join(_dir, '{}_dm'.format(self.exp_name)))
        torch.save(model.state_dict(), os.path.join(_dir, '{}-ckpt.pth'.format(self.exp_name)))
        pickle.dump(losses, open(os.path.join(_dir, '{}-tc.pkl'.format(self.exp_name)), 'wb'))

    def eval(self, model, datamodule, visualization=None):
        model = model.to(self.device)
        _dir = os.path.join(self.checkpoint_dir, self.exp_type)
        PATH = os.path.join(_dir, '{}-ckpt.pth'.format(self.exp_name))
        # load the trained forward model 
        model.load_state_dict(torch.load(PATH))
        model = model.eval()

        all_params = datamodule.test_dataset.tensors[0]
        all_targets = datamodule.test_dataset.tensors[1]
        preds, _ = model(all_params)
        rand_preds = torch.rand(preds.shape)

        print("Check: ", all_targets)
        print(torch.sum(all_targets, dim=1))
        print(preds)
        print(torch.sum(preds, dim=1))
        print(rand_preds)
        print(torch.sum(rand_preds, dim=1))
        
        err = torch.abs(preds - all_targets)
        rand_err = torch.abs(rand_preds - all_targets)

        _err = err.detach().cpu().numpy()
        _randerr = rand_err.detach().cpu().numpy()
        
        ##### 240124
        # Returning stim info
        stim_params = all_params.detach().cpu().numpy()
        target_outcomes = all_targets.detach().cpu().numpy()
        
        print()
        print("Stim parameters (stim_params in code): ", stim_params.shape)
        print("Target outcomes (target_outcomes in code): ", target_outcomes.shape)        
        
        # Printing and saving error
        file_path = "temp.json"

        print()
        print("Errors were saved in " + file_path)
        print("Error - mean (L1 loss): ", np.mean(_err))
        print("Error - muscle: ", np.mean(_err, axis=0))
        err_dict = {
            'err_total': _err.tolist(),
            'err_muscle': np.mean(_err, axis=0).tolist(),
            'err_mean': np.mean(_err).tolist()
        }

        with open(file_path, 'w') as json_file:
            json.dump(err_dict, json_file)
            
        # Providing dummy input to model
        dummy_input = torch.tensor(np.array([0.3, 0.6, 0.4, 0.8], dtype=np.float32)).to(self.device)
        preds_dummy, _ = model(dummy_input)
        
        print()
        print("Provided dummy input: ", dummy_input)
        print("Model prediction for dummy input: ", preds_dummy)
        #####
        
        X = {'channels': [x for x in datamodule.selected_muscles], 'err': _err, 'randerr': _randerr}
        pickle.dump(X, open(os.path.join('ckpts', 'StimToEMG', '{}_l1_errors.p '.format(self.exp_name)), 'wb'))

    def plot_electrode_array(self, theta, gt, target, json_name='FlorenceXY', savename='fullposterior.png'):
        js = json.load(open(os.path.join('data', '{}.json'.format(json_name)), 'r'))
        
        fig = plt.figure(figsize=(12,6))
        grid = plt.GridSpec(4, 9, hspace=0.2, wspace=0.2)
 
        ax = fig.add_subplot(grid[1:,:4]) #fig.add_subplot(121)

        for item in js['Caudal']+js['Rostral']:
            if not (type(item['Norm_Points']) is dict):
                continue
            xy_bl = item['Norm_Points']['Bottom_Left']
            xy_br = item['Norm_Points']['Bottom_Right']
            xy_tr = item['Norm_Points']['Top_Right']
            xy_tl = item['Norm_Points']['Top_Left']
            ax.plot([xy_bl[0], xy_br[0], xy_tr[0], xy_tl[0], xy_bl[0]], [xy_bl[1], xy_br[1], xy_tr[1], xy_tl[1], xy_bl[1]], c='tab:gray')

        #ax.axis('equal')
        ax.axis('off')

        ax.scatter(theta[:,2], theta[:,3], s=1, c='b', marker='.', alpha=0.75, label='posterior samples')
        ax.scatter(gt[:,2], gt[:,3], s=10, c='r', marker='*', label='groundtruth electrode')
        #ax.legend()

        ax = fig.add_subplot(grid[0, :4])
        ax.imshow(target, vmin=0., vmax=1., cmap='Blues')
        #ax.axis('off')
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels(['L-PL', 'L-GA', 'L-GR', 'L-BF', 'R-PL', 'R-GA', 'R-GR', 'R-BF'])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

        #####
        # plot (f,a) axis
        #####
        N = 100
        bins = np.linspace(0, 1, N)
        
        # Set up the axes with gridspec
        #fig = plt.figure(figsize=(6, 6))
        #grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
        
        main_ax = fig.add_subplot(grid[1:, 5:-1])
        y_hist = fig.add_subplot(grid[1:, -1])
        x_hist = fig.add_subplot(grid[0, 5:-1])

        X, Y = np.mgrid[0:1:100j, 0:1:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([theta[:,1].numpy(), theta[:, 0].numpy()])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)

        # Covariance
        main_ax.imshow(np.rot90(Z), cmap=plt.cm.Oranges, extent=[0., 1., 0., 1.], zorder=1)
        main_ax.spines['right'].set_visible(False)
        main_ax.spines['top'].set_visible(False)
        main_ax.yaxis.set_ticks_position('left')
        main_ax.xaxis.set_ticks_position('bottom')
        main_ax.set_xticks([0., 1.])
        main_ax.set_xticklabels([0., 1.])

        main_ax.set_yticks([0., 1.])
        main_ax.set_yticklabels([0., 1.])

        main_ax.set_xlabel('normalized amplitude')
        main_ax.set_ylabel('normalized frequency')
 
        # Marginals on the attached axes
        x_hist.hist( theta[:,1].numpy(), density=True, bins=bins, alpha=0.75, color='tab:brown', orientation='vertical')
        kde = stats.gaussian_kde(theta[:,1].numpy())
        x_hist.spines['right'].set_visible(False)
        x_hist.spines['top'].set_visible(False)
        x_hist.spines['left'].set_visible(False)
        x_hist.set_xticks([])
        x_hist.set_yticks([])

        y_hist.hist(theta[:,0].numpy(), density=True, bins=bins, color='tab:brown', alpha=0.75, orientation='horizontal')
        kde = stats.gaussian_kde(theta[:,0].numpy())
        y_hist.spines['top'].set_visible(False)
        y_hist.spines['bottom'].set_visible(False)
        y_hist.spines['right'].set_visible(False)
        y_hist.set_xticks([])
        y_hist.set_yticks([])

        plt.savefig(savename , bbox_inches='tight')
        #plt.show()
      
 
    def inference(self, model, datamodule, electrode_index, target_index, train_nde=True, model_save_path=None): ##### 230425: train_nde=False
        model = model.to(self.device)
        all_params = datamodule.test_dataset.tensors[0]
        all_targets = datamodule.test_dataset.tensors[1]
        elec_encoding = datamodule.elec_encoding

        print('Of {} available targets, selecting index {}'.format(all_targets.shape[0], target_index))
        
        _dir = os.path.join(self.checkpoint_dir, self.exp_type)
        PATH = os.path.join(_dir, '{}-ckpt.pth'.format(self.exp_name))
        print("Loading trained model: ", PATH)
        # load the trained forward model 
        model.load_state_dict(torch.load(PATH))
        #model = model.eval()

        target = all_targets[target_index]
        parameters = all_params[target_index]

        simulator = GlobalSimulator(model, device=self.device)

        inference = Inference(
            elec_encoding='pos',
            num_rounds=self.num_rounds, 
            num_simulations=self.num_simulations, 
            simulation_batch_size=self.simulation_batch_size,
            training_batch_size=self.training_batch_size,
            num_samples=self.num_samples,
            filtering_ratio=self.filtering_ratio,
            num_proposals=self.num_proposals,
            timeout=self.timeout
        )

        _pdir = os.path.join(_dir, self.exp_name)
        if not os.path.exists(_pdir):
            os.makedirs(_pdir)
        
        if train_nde:
            start = time.time()
            posterior = inference.train(simulator, target, _xy=None, height=None, width=None)
            torch.save(posterior, os.path.join(_pdir, 'GlobalXY_HD64Retest.pth'))
            end = time.time()
            print('elapsed time: %.1f sec' % (end - start))
        else:
            posterior = torch.load(os.path.join(_pdir, 'GlobalXY_HD64Retest.pth')) 

        if posterior is None:
            # finish the inference process when it is timeout
            return

        print("A", _pdir)
        SPATH = os.path.join(_pdir, 'EI_{}_TI_{}.png'.format(electrode_index, target_index))
        # plot posterior samples
        print("B", target.shape, parameters.shape)
        print("C", posterior) #.shape, target.shape, parameters.shape)
        
        import signal ##### 231024
        # change the behavior of SIGALRM
        signal.signal(signal.SIGALRM, timeout_handler)
        # set timeout
        signal.alarm(self.timeout)
        
        try: ##### 231024
            fig, theta = inference.pairplot(posterior, target, parameters)
            fig.savefig(SPATH)

            PPATH = os.path.join(_pdir, 'EI_{}_TI_{}_graphic.png'.format(electrode_index, target_index))
            self.plot_electrode_array(theta=theta, gt=parameters, target=target, json_name='FlorenceXY', savename=PPATH)

            x, theta, log_probability = inference.sampling_proposals(simulator, posterior, target)

            x, theta, log_probability, dist = inference.filtering_proposals(x, target, theta, log_probability, metric='l1')

            # save proposals as JSON file
            gt_ees = parameters.cpu().numpy()
            gt_ees = datamodule.inverse_transform_ees(gt_ees) #[None, :]

            x = x.cpu().numpy()
            theta = theta.cpu().numpy()
            theta = datamodule.inverse_transform_ees(theta)
            if elec_encoding == 'onehot':
                proposed_electrode = datamodule.transformers['ees']['parameterization'].idx2elec[2**electrode_index].item()

            proposals = OrderedDict()
            proposals['gt_emg'] = target.cpu().numpy().tolist()
            proposals['gt_ees'] = {
                'freq': gt_ees['freq'][0].item(),
                'amp': gt_ees['amp'][0].item(),
                'elec': gt_ees['elec'][0].tolist() if elec_encoding=='onehot' else gt_ees['elec'][0].tolist()
                #'elec': gt_ees['elec'][0].tolist() if elec_encoding=='onehot' else xy2idx(gt_ees['elec'][0].tolist())
            }

            for n in range(x.shape[0]):
                proposals['proposal_%d' % n] = {
                    'emg': x[n].tolist(),            
                    'ees': {
                        'freq': theta['freq'][n].item(),
                        'amp': theta['amp'][n].item(),
                        'elec': proposed_electrode if elec_encoding=='onehot' else theta['elec'][n].tolist() #same change here
                    },
                    'log_probability': log_probability[n].item(),
                    'dist': dist[n].item()
                }

            # write JSON
            with open(os.path.join(_pdir, 'proposals_960_T_{}.json'.format(target_index[0])), 'w', encoding="utf-8") as fp:
                json.dump(proposals, fp, ensure_ascii=False, indent="\t")
                
        except TimeoutException: ##### 231024
            print('Inference terminated due to timeout...')
            pass
