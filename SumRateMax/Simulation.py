import PowerCtrlNet as p_net
import tensorflow as tf
import numpy as np
import datetime
import ComLib as clb
import os

def max_and_argmax(rate_matrix, random_perm=False):
    # random_perm is true: randomly select one network if some have the same results
    batch_size = np.size(rate_matrix, 1)
    model_num = np.size(rate_matrix, 0)
    max_rate = np.zeros([batch_size])
    max_rate_id = np.zeros([batch_size], np.int32)
    model_num_maximum_rate = np.zeros([batch_size], np.int32)  # number of models yielding the maximum rate
    for i in range(batch_size):
        rate = rate_matrix[:, i]
        if random_perm:
            perm_id = np.random.permutation(model_num)
            rate = rate[perm_id]

        max_rate[i] = np.max(rate)
        if random_perm:
            max_rate_id[i] = perm_id[np.argmax(rate)]
        else:
            max_rate_id[i] = np.argmax(rate)
        model_num_maximum_rate[i] = np.sum(np.int32(np.abs(rate - np.max(rate)) < 1e-10))
    return max_rate, max_rate_id, model_num_maximum_rate


def simulation_combine_dnns(top_config, model_id_set):
    p_ctrl_net = {}
    sum_rate = {}
    model_num = np.size(model_id_set)
    for i in range(model_num):
        p_ctrl_net[i] = p_net.PowerCtrlNet(top_config)
        p_ctrl_net[i].build_network(model_id_set[i])
        sum_rate[i] = p_ctrl_net[i].get_sum_rate_vector(p_ctrl_net[i].net_out, p_ctrl_net[i].net_in, top_config.simu_noise_power, top_config.binary_power_ctrl_enabled)

    # set the cpu number used. debug
    # cpu_num = int(os.environ.get('CPU_NUM', 1))
    # config = tf.ConfigProto(device_count={"CPU": cpu_num},
    #                         inter_op_parallelism_threads=cpu_num,
    #                         intra_op_parallelism_threads=cpu_num,
    #                         log_device_placement=True)
    # sess = tf.Session(config=config)
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    for i in range(model_num):
        p_ctrl_net[i].restore_network(sess, model_id_set[i])

    batch_size = top_config.simu_batch_size
    batch_num = int(top_config.simutimes / batch_size)
    rng = np.random.RandomState(top_config.simu_seed)
    sum_rate_v = 0.0

    if top_config.output_all_rates_to_file:
        out_file = format("%s/DnnSumRate_BinaryPowerCtrl%d_model(%d_%d).dat" % (top_config.base_folder, top_config.binary_power_ctrl_enabled, model_id_set[0],
                                                                                model_id_set[model_num-1]))
        fout = open(out_file, "wb")

    output_abs_H = False
    if output_abs_H:
        fout_abs_H = open("ChannelCoeffs.dat", "wb")

    rate_matrix = np.zeros([model_num, batch_size])

    start_time = datetime.datetime.now()
    for bid in range(batch_num):
        # if np.mod(bid, 10) == 0:
        #     print("Batch %d" % bid)

        batch_abs_H = clb.generate_CSI(top_config.user_num, batch_size, rng, top_config.csi_distribution)
        if output_abs_H:
            batch_abs_H.astype(np.float32).tofile(fout_abs_H)

        for i in range(model_num):
            rate_matrix[i,:] = sess.run(sum_rate[i], feed_dict={p_ctrl_net[i].net_in: batch_abs_H})

        # calculate the frequency one model is selected
        sum_rate_vector_max, max_rate_idx, model_num_maximum_rate = max_and_argmax(rate_matrix)

        if top_config.output_all_rates_to_file:
            sum_rate_vector_max = sum_rate_vector_max.astype(np.float32)
            sum_rate_vector_max.tofile(fout)

        sum_rate_v = sum_rate_v + np.mean(sum_rate_vector_max)

    end_time = datetime.datetime.now()

    sum_rate_v /= float(batch_num)
    sess.close()
    if top_config.output_all_rates_to_file:
        fout.close()
    if output_abs_H:
        fout_abs_H.close()
    print("Average sum rate: %.4f" % sum_rate_v)
    print("Used time %ds." % (end_time-start_time).seconds)

    return sum_rate_v

def simulation_combine_dnns_numpy(top_config, model_id_set):
    p_ctrl_net = {}
    sum_rate = {}
    model_num = np.size(model_id_set)
    for i in range(model_num):
        p_ctrl_net[i] = p_net.PowerCtrlNet(top_config)
        p_ctrl_net[i].build_network(model_id_set[i])

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    for i in range(model_num):
        p_ctrl_net[i].restore_network(sess, model_id_set[i])

    ## extract network to numpy
    weights = {}
    bias = {}
    best_means = {}
    best_std_variances = {}
    best_scale = {}
    best_offset = {}
    for layer in range(top_config.layer_num):
        weights[layer] = np.array(sess.run(p_ctrl_net[0].weights[layer]))
        bias[layer] = np.array(sess.run(p_ctrl_net[0].biases[layer]))
        if layer != top_config.layer_num - 1:
            best_means[layer] = np.array(sess.run(p_ctrl_net[0].best_means[layer]))
            best_std_variances[layer] = np.array(np.sqrt(sess.run(p_ctrl_net[0].best_variances[layer])))
            best_scale[layer] = np.array(sess.run(p_ctrl_net[0].scaling_factors[layer]))
            best_offset[layer] = np.array(sess.run(p_ctrl_net[0].offsets[layer]))

    sess.close()

    rng = np.random.RandomState(top_config.simu_seed)
    sum_rate_v = 0.0

    start_time = datetime.datetime.now()
    for bid in range(top_config.simutimes):
        abs_H = clb.generate_CSI(top_config.user_num, 1, rng, top_config.csi_distribution)
        layer_out = abs_H
        for layer in range(top_config.layer_num):
            layer_in = layer_out
            layer_out = np.matmul(layer_in, weights[layer]) + bias[layer]
            if layer!=top_config.layer_num-1:
                layer_out = (layer_out - best_means[layer]) / best_std_variances[layer] * best_scale[layer] + best_offset[layer]
                layer_out[layer_out < 0] = 0
            else:
                layer_out = 1 / (1 + np.exp(-layer_out))

        sum_rate_v += clb.calc_sum_rate(layer_out, abs_H, top_config.noise_power, True)


    end_time = datetime.datetime.now()

    sum_rate_v /= float(top_config.simutimes)
    print("Average sum rate: %.4f" % sum_rate_v)
    print("Used time %ds." % (end_time-start_time).seconds)

    return sum_rate_v
