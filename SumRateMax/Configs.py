import numpy as np
class TopConfig:
    def __init__(self, argv=None):
        self.function = "Simulation"  #  Simulation or Train
        # communication
        self.user_num = 10  # number of users, currently, if it is set to not 20 or 10, the node_nums have also been reset.
        self.tx_snr = 10
        self.csi_distribution = "Rayleigh" #""Rayleigh" "Rice" or "Geometry"
        self.noise_power = 10**(-self.tx_snr/10.0)
        self.simu_tx_snr = self.tx_snr
        self.simu_noise_power = 10**(-self.simu_tx_snr/10.0)  ## Maximum transmission power is normalized to 1 in this project.

        # network architecture
        self.input_len = self.user_num ** 2
        # FCN configurations
        self.layer_num = 3
        self.node_nums = np.array([0, 0, 0])
        if self.user_num == 20:
            self.node_nums = np.array([400, 200, self.user_num], np.int32)  # node number does not contain the input layer
        elif self.user_num == 10:
            self.node_nums = np.array([200, 100, self.user_num], np.int32)

        self.net_input_format = "H"   # H means the network input is the absolute value of the channel coefficients,
                                      # H2 means it's the square of the channel coeffs

        # training
        self.model_id = 0
        self.load_data_from_files = False
        self.flag_restore_network = False

        self.training_sample_num = 1000000  # number of training samples, this is only valid when the training data is generated offline.
        self.test_sample_num = 10000  # number of testing  samples
        self.training_mini_batch_size = 1000  # training minibach size
        self.test_mini_batch_size = self.test_sample_num
        self.epoch_num = int(1e5)  # number of training epochs
        self.check_interval = 50
        self.train_seed = 1  # set random seed for training set
        self.test_seed = 0  # set random seed for test set
        self.initialization = "Xavier"  # "Xavier" or "TN" (Truncated normal)


        # simulation
        self.simu_seed = 100
        self.simutimes = 20000
        self.simu_batch_size = 2000
        self.simu_net_id_set = range(0, 1)
        self.output_all_rates_to_file = False
        self.binary_power_ctrl_enabled = True

        if argv is not None and len(argv) > 1:
            self.parse_cmd_line(argv)

        self.gen_base_folder()


    def gen_base_folder(self):
        node_num_str = np.array2string(self.node_nums, separator='_', formatter={'int': lambda d: "%d" % d})
        node_num_str = node_num_str[1:len(node_num_str) - 1]
        self.base_folder = format('./SavedModels/UserNum%d_TxSNR%.1fdB_%s/FCN(%s)_MiniBatch%d_Epoch%d' % (self.user_num, self.tx_snr,
                                                                                                              self.csi_distribution, node_num_str,
                                                                                                           self.training_mini_batch_size, self.epoch_num))

        self.data_folder = format('./Data')  # this is only valid if flag_load_model_init_from_file==True

    def parse_cmd_line(self, argv):
        if len(argv) == 1:
            return
        id = 1
        while id < len(argv):
            if argv[id]=='-TxSNR':
                self.tx_snr = float(argv[id+1])
                self.noise_power = 10 ** (-self.tx_snr / 10.0)
                self.simu_tx_snr = self.tx_snr
                self.simu_noise_power = 10 ** (-self.simu_tx_snr / 10.0)
            elif argv[id]=='-UserNum':
                self.user_num = int(argv[id+1])
                self.input_len = self.user_num**2
                if self.user_num == 20:
                    self.node_nums = np.array([400, 200, self.user_num], np.int32)  # node number does not contains the input layer
                elif self.user_num == 10:
                    self.node_nums = np.array([200, 100, self.user_num], np.int32)

            elif argv[id]=='-Func':
                self.function = argv[id+1]
            elif argv[id]=='-BinaryPwrCtrl':
                self.binary_power_ctrl_enabled = (argv[id+1]=="True")
            elif argv[id]=='-TrainSeed':
                self.train_seed = int(argv[id+1])
                self.test_seed = self.train_seed - 1
            elif argv[id]=='-EpochNum':
                self.epoch_num = int(argv[id+1])
            elif argv[id]=='-ModelId':
                self.model_id = int(argv[id+1])
            elif argv[id]=='-NetInput':
                self.net_input_format = argv[id+1]
            elif argv[id]=='-NetInit':
                self.initialization = argv[id+1]
            else:
                print("Invalid paras!")
                exit(0)
            id = id + 2