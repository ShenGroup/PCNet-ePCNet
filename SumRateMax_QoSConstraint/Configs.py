
import numpy as np
class TopConfig:
    def __init__(self, argv=None):
        self.function = "Simulation"  #  Simulation or Train or GenData
        # communication configurations
        self.user_num = 5  # number of users
        self.tx_snr = 10
        self.min_rates = np.zeros([self.user_num])
        self.min_rates[0] = 0.5
        self.min_rates[1] = 0.5
        self.min_rates[2] = 0.0
        self.min_rates[3] = 0.0
        self.min_rates[4] = 0.0
        self.noise_power = 10**(-self.tx_snr/10.0)
        self.simu_tx_snr = self.tx_snr
        self.simu_noise_power = 10**(-self.simu_tx_snr/10.0)  ## Maximum transmission power is normalized to 1 in this project.

        # network architecture
        self.input_len = self.user_num**2
        self.layer_num = 3
        if self.user_num == 20:
            self.node_nums = np.array([400, 200, self.user_num], np.int32) # node number does not contain the input layer
        elif self.user_num == 10:
            self.node_nums = np.array([200, 100, self.user_num], np.int32)
        elif self.user_num == 5:
            self.node_nums = np.array([50, 25, self.user_num], np.int32)

        self.net_input_format = "H"   # H means the network input is the absolute value of the channel coefficients, H2 means it's the square of the channel coeffs

        # training
        self.model_id = 0
        self.load_data_from_files = True
        self.flag_restore_network = False

        self.training_sample_num = 1000000  # number of training samples, this is only valid when the training data is generated offline.
        self.test_sample_num = 10000  # number of testing samples
        self.training_mini_batch_size = 1000  # training minibach size
        self.test_mini_batch_size = 10000
        self.epoch_num = 100000  # number of training epochs
        self.check_interval = 50
        self.train_seed = 1  # set random seed for training set
        self.test_seed = 0  # set random seed for test set
        self.constraint_loss_scale = 10

        # simulation
        self.simu_seed = 100
        self.simutimes = 10000
        self.simu_batch_size = 10000
        self.simu_net_id_set = range(0, 2)
        self.output_all_rates_to_file = False
        self.binary_power_ctrl_enabled = False

        if argv is not None and len(argv) > 1:
            self.parse_cmd_line(argv)

        self.gen_base_folder()


    def gen_base_folder(self):
        self.base_folder = format('./Models')
        self.data_folder = format('./Data')

    def parse_cmd_line(self, argv):
        if len(argv) == 1:
            return
        id = 1
        while id < len(argv):
            if argv[id]=='-TxSNR':
                self.tx_snr = float(argv[id+1])
            elif argv[id]=='-UserNum':
                self.user_num = int(argv[id+1])
                self.input_len = self.user_num**2
                self.node_nums[np.size(self.node_nums)-1] = self.user_num
            elif argv[id]=='-Func':
                self.function = argv[id+1]
            elif argv[id]=='-BinaryPwrCtrl':
                self.binary_power_ctrl_enabled = (argv[id+1]=="True")
            elif argv[id]=='-TrainSeed':
                self.train_seed = int(argv[id+1])
            elif argv[id]=='-EpochNum':
                self.epoch_num = int(argv[id+1])
            elif argv[id]=='-ModelId':
                self.model_id = int(argv[id+1])
            elif argv[id]=='-ConstraintLoss':
                self.constraint_loss_scale = float(argv[id+1])
            elif argv[id]=='-SimuSeed':
                self.simu_seed = int(argv[id + 1])
            elif argv[id]=='-NodeNum':
                self.node_nums = np.fromstring(argv[id+1], np.int32, sep='_')
            else:
                print("Invalid paras!")
                exit(0)
            id = id + 2