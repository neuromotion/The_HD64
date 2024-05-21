class Config():
    def __init__(self):
        self.seed = 1234
        self.ckpt_dir = 'ckpts'    
        self.data_dir = 'data'
        
        '''Specify which EMG channels to select
        Exp: Stim2EMG
        '''
        # for retest session 1
#         self.selected_muscles = [1, 2, 5, 7, 8, 10]
#         self.selected_muscles = [3, 4, 6, 8, 10, 11, 12, 13]
#         self.selected_muscles = [4, 8, 10, 11, 12, 13]
        self.selected_muscles = [1, 2, 3, 4, 5, 6]    

        # for retest session 2
        #self.selected_muscles = [2, 5, 7, 8, 10, 11, 12, 13]

        self.ignore_electrodes = [] # in case we decide some electrodes are bad

        '''Specify the taxonamy for grouping the dermatome
        Exp: Stim2Sense
        '''
        '''
        self.selected_bodyparts = [
                            [x for x in range(0, 500)],
                            [x for x in range(500, 1000)],
                            [x for x in range(1000, 1500)]
                        ]
        '''
        self.selected_bodyparts = [[x for x in range(k*100,(k+1)*100)] for k in [15, 18, 22, 23]]

        '''Specify the class labels to decode
        Exp: Int2LFP
        '''
        self.selected_intentions = ['Left', 'Right', 'None']
        self.selected_channels = [x for x in range(129, 161)]
        self.selected_dur = ['0', '0_5'] #['x0_00013333', 'x0_29987']
 
#         self.elec_encoding = 'onehot' # 'pos'
        self.elec_encoding = 'pos'


        self.holdout_frequencies = [50.]
        self.holdout_amplitudes = [1000.] # 900 for retest session 1

        # training hyperparameters
        self.stim2emg = {
                    'input_dims': 4,
                    'output_dims': 6, # 6 for retest session 1; 8 for retest session 2
                    'lr': 0.003,
                    'weight_decay': 0.0005
                    }

        self.stim2sen = {
                    'input_dims': 8, # f + a + num_electrodes 
                    'output_dims': 4,
                    'lr': 0.003,
                    'weight_decay': 0.0005
                    }


        self.num_epochs = 1000 #1000 
        self.deterministic = True
        self.eval_interval = 100
        self.train_bs = None
        self.valid_bs = None
        self.test_bs = None
        self.device = 'cpu'

        # inference hyperparameters
        self.num_rounds = 2
        self.num_simulations = 1024*128
        self.simulation_batch_size = 4096*4 
        self.training_batch_size = 1024*32 
        self.num_samples = 10000
        self.filtering_ratio = 0.1
        self.num_proposals = 10
        self.timeout = 500
