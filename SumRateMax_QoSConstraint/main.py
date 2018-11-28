### important
# The element h_{i,j} of the CSI matrix denotes the channel state from transmitter i to receiver j.
###

import Configs as cfg
import PowerCtrlNet as p_net
import Simulation as simu
import sys
import numpy as np

top_config = cfg.TopConfig(sys.argv)

if top_config.function == "GenData":
    p_net.generate_train_data(top_config, "Train")
    p_net.generate_train_data(top_config, "Test")
elif top_config.function == "Train":
    p_network = p_net.PowerCtrlNet(top_config)
    p_network.train_network(top_config.model_id)
elif top_config.function == "Simulation":
    simu.simulation_combine_dnns(top_config, top_config.simu_net_id_set)