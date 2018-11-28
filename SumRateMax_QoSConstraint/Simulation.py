import PowerCtrlNet as p_net
import tensorflow as tf
import numpy as np
import datetime
import ComLib as opc

def max_and_argmax(rate_matrix):
    batch_size = np.size(rate_matrix, 1)
    model_num = np.size(rate_matrix, 0)
    max_rate = np.zeros([batch_size])
    max_rate_id = np.zeros([batch_size], np.int32)
    model_num_maximum_rate = np.zeros([batch_size], np.int32) # number of models yielding the maximum rate
    for i in range(batch_size):
        rate = rate_matrix[:, i]
        perm_id = np.random.permutation(model_num)
        rate = rate[perm_id]
        max_rate[i] = np.max(rate)
        max_rate_id[i] = perm_id[np.argmax(rate)]
        model_num_maximum_rate[i] = np.sum(np.int32(np.abs(rate - np.max(rate)) < 1e-7))
    return max_rate, max_rate_id, model_num_maximum_rate


def simulation_combine_dnns(top_config, model_id_set):
    p_ctrl_net = {}
    per_user_rate = {}
    model_num = np.size(model_id_set)
    for i in range(model_num):
        p_ctrl_net[i] = p_net.PowerCtrlNet(top_config)
        p_ctrl_net[i].build_network(model_id_set[i])
        per_user_rate[i] = p_ctrl_net[i].get_per_user_rate(p_ctrl_net[i].net_out, p_ctrl_net[i].net_in, top_config.simu_noise_power, top_config.binary_power_ctrl_enabled)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(model_num):
        p_ctrl_net[i].restore_network(sess, model_id_set[i])

    batch_size = top_config.simu_batch_size
    rng = np.random.RandomState(top_config.simu_seed)
    sum_rate_v = 0.0

    if top_config.output_all_rates_to_file:
        out_file = format("%s/DnnSumRate_BinaryPowerCtrl%d.dat" % (top_config.base_folder, top_config.binary_power_ctrl_enabled))
        fout = open(out_file, "wb")

    output_abs_H = False
    if output_abs_H:
        fout_abs_H = open("ChannelCoeffsForSimulation.dat", "wb")

    start_time = datetime.datetime.now()

    valid_simutimes = 0
    simutimes_ensemble_out_feasible_sln = 0
    while valid_simutimes < top_config.simutimes:
        batch_abs_H = opc.generate_CSI(top_config.user_num, batch_size, rng)
        feasible_flag, threshold_power = opc.check_feasibility(np.square(np.reshape(batch_abs_H, [batch_size, top_config.user_num, top_config.user_num])), 2**top_config.min_rates
                                                             - 1, top_config.simu_noise_power)
        batch_size_new = np.sum(feasible_flag)
        batch_abs_H = batch_abs_H[feasible_flag, :]
        threshold_power = threshold_power[feasible_flag, :]
        rate_matrix = np.zeros([model_num, batch_size_new])
        if np.size(np.shape(batch_abs_H)) == 1:
            batch_abs_H = np.reshape(batch_abs_H, [1, -1])

        if output_abs_H:
            batch_abs_H.astype(np.float32).tofile(fout_abs_H)

        user_rate_feasible_flag_ensemble = np.zeros([batch_size_new], dtype=np.bool)  # indicate whether the ensemble outputs a feasible sln.
        for i in range(model_num):
            user_rate, power = sess.run([per_user_rate[i], p_ctrl_net[i].net_out], feed_dict={p_ctrl_net[i].net_in: batch_abs_H})

            user_rate_feasible_flag = np.sum(user_rate < top_config.min_rates - 1e-6, axis=1) == 0
            user_rate[~user_rate_feasible_flag, :] = top_config.min_rates
            ######################################################################################################################################
            # for those unfeasible output, directly use the power obtained by normalizing the power solution of the minimum power usage
            undecided_abs_H = batch_abs_H[~user_rate_feasible_flag, :]
            power_undecided_abs_H = threshold_power[~user_rate_feasible_flag, :]
            undecided_abs_H = np.reshape(undecided_abs_H, [-1, top_config.user_num**2])
            power_undecided_abs_H = np.reshape(power_undecided_abs_H, [-1, top_config.user_num])
            power_undecided_abs_H = opc.normalize_power(power_undecided_abs_H, 1.0)
            user_rate_undecided_H = opc.calc_per_user_rate(power_undecided_abs_H, undecided_abs_H, top_config.noise_power, False)
            user_rate[~user_rate_feasible_flag, :] = user_rate_undecided_H

            ######################################################################################################################################

            user_rate_feasible_flag_ensemble += user_rate_feasible_flag  # indicate whether the ensemble outputs a feasible sln.
            rate_matrix[i, :] = np.sum(user_rate, axis=1)

        # calculate the frequency one model is selected
        sum_rate_vector_max, max_rate_idx, model_num_maximum_rate = max_and_argmax(rate_matrix)

        if top_config.output_all_rates_to_file:
            sum_rate_vector_max = sum_rate_vector_max.astype(np.float32)
            sum_rate_vector_max.tofile(fout)

        # to make the valid simutimes exactly equal to the target
        if (valid_simutimes + batch_size_new > top_config.simutimes):
            sum_rate_v = sum_rate_v + np.sum(sum_rate_vector_max[0:top_config.simutimes-valid_simutimes])
            batch_size_new = top_config.simutimes-valid_simutimes
        else:
            sum_rate_v = sum_rate_v + np.sum(sum_rate_vector_max)

        # calculate the simutimes that the ensemble outputs a feasible sln.
        simutimes_ensemble_out_feasible_sln += np.sum(user_rate_feasible_flag_ensemble[0:batch_size_new])

        valid_simutimes += batch_size_new
    sum_rate_v /= float(valid_simutimes)

    sess.close()
    if top_config.output_all_rates_to_file:
        fout.close()

    if output_abs_H:
        fout_abs_H.close()

    print("Average sum rate: %.4f" % sum_rate_v)
    print("Hit rate the ensemble outputs a feasible sln: %.4f" % (simutimes_ensemble_out_feasible_sln / float(valid_simutimes)))
    print("Used time %ds." % (datetime.datetime.now()-start_time).seconds)
    return sum_rate_v, (simutimes_ensemble_out_feasible_sln / float(valid_simutimes))