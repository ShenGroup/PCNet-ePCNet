
import tensorflow as tf
import numpy as np
import datetime
import os
import ComLib as opc

def generate_train_data(top_config, goal):
    start_time = datetime.datetime.now()
    if goal == "Train":
        rng = np.random.RandomState(top_config.train_seed)
        batch_size = top_config.training_mini_batch_size
        sample_num = top_config.training_sample_num
        fout = open(format('%s/TrainingData.dat' % (top_config.data_folder)), "wb")
    elif goal == "Test":
        rng = np.random.RandomState(top_config.test_seed)
        batch_size = top_config.test_mini_batch_size
        sample_num = top_config.test_sample_num
        fout = open(format('%s/TestData.dat' % (top_config.data_folder)), "wb")
    else:
        print("Invalid goal.")
        exit(0)

    K = top_config.user_num
    valid_data_sample_num = 0
    while (valid_data_sample_num < sample_num):
        batch_abs_H = opc.generate_CSI(K, batch_size, rng)
        feasible_flag, _ = opc.check_feasibility(np.reshape(np.square(batch_abs_H), [batch_size, K, K]), 2**top_config.min_rates - 1, top_config.noise_power)
        num_feasible_sample = np.sum(feasible_flag)
        if num_feasible_sample == 0:
            continue

        batch_abs_H = batch_abs_H[feasible_flag, :]
        batch_abs_H = batch_abs_H.astype(np.float32)
        batch_abs_H.tofile(fout)
        valid_data_sample_num += num_feasible_sample

    fout.close()
    print("Used time %ds." % (datetime.datetime.now() - start_time).seconds)


class DataIO:
    def __init__(self, top_config):
        print("Construct a data IO class!\n")
        self.top_config = top_config
        self.load_data_from_files = top_config.load_data_from_files
        if self.load_data_from_files:
            self.fin_train = open(format('%s/TrainingData.dat' % (top_config.data_folder)), "rb")
            self.fin_test = open(format('%s/TestData.dat' % (top_config.data_folder)), "rb")

        self.rng_training = np.random.RandomState(self.top_config.train_seed)
        self.rng_test = np.random.RandomState(self.top_config.test_seed)
        self.noise_power = top_config.noise_power
        self.all_training_data = None
        self.all_test_data = None
        self.test_data_pos = 0

    def __del__(self):
        if self.load_data_from_files:
            self.fin_train.close()
            self.fin_test.close()

    def gen_a_mini_batch(self, mini_batch_size, rng, goal):
        K = self.top_config.user_num
        if self.load_data_from_files:
            if goal=="Training":
                sample_num = self.top_config.training_sample_num
                fin = self.fin_train
            elif goal=="Test":
                sample_num = self.top_config.test_sample_num
                fin = self.fin_test
            sample_id = np.random.randint(sample_num - mini_batch_size + 1)
            fin.seek(K**2 * 4 * sample_id)
            batch_net_in = np.fromfile(fin, np.float32, mini_batch_size *  K**2)
            batch_net_in = np.reshape(batch_net_in, [mini_batch_size, K ** 2])
        else:
            CH = 1 / np.sqrt(2) * (rng.randn(mini_batch_size, K ** 2) + 1j * rng.randn(mini_batch_size, K ** 2))
            batch_net_in = np.abs(CH)
        return batch_net_in


    def gen_a_training_mini_batch(self, mini_batch_size):
        ###########################################################################################################################
        ## generate a mini_batch from the buffer
        if self.top_config.load_data_from_files:
            if self.all_training_data is None:
                self.all_training_data = self.gen_a_mini_batch(self.top_config.training_sample_num, self.rng_training, "Training")
            sample_id = np.random.randint(self.top_config.training_sample_num - mini_batch_size + 1)
            batch_net_in = self.all_training_data[sample_id:(sample_id + mini_batch_size), :]
        ###########################################################################################################################

        return batch_net_in

    def gen_a_test_mini_batch(self):
        mini_batch_size = self.top_config.test_mini_batch_size
        if self.all_test_data is None:
            self.all_test_data = self.gen_a_mini_batch(self.top_config.test_sample_num, self.rng_test, "Test")

        batch_net_in = self.all_test_data[self.test_data_pos:(self.test_data_pos + mini_batch_size), :]
        self.test_data_pos = self.test_data_pos + mini_batch_size
        if self.test_data_pos >= self.top_config.test_sample_num:
            self.test_data_pos = 0
        return batch_net_in

    def load_all_training_data(self):
        if self.all_training_data is None:
            self.all_training_data = self.gen_a_mini_batch(self.top_config.training_sample_num, self.rng_training, "Training")
        return self.all_training_data


    def load_all_test_data(self):
        if self.all_test_data is None:
            self.all_test_data = self.gen_a_mini_batch(self.top_config.test_sample_num, self.rng_test, "Test")
        return self.all_test_data



class PowerCtrlNet:
    def __init__(self, top_config):
        self.top_config = top_config
        # network parameters
        self.weights = {}
        self.biases = {}
        self.best_weights = {}
        self.best_biases = {}
        self.assign_best_weights = None
        self.assign_best_biases = None
        self.assign_net_in_train = None
        # net input and output
        self.net_in = None
        self.net_in_train = None
        self.net_in_test = None
        self.net_out_before_sigmoid = None
        self.net_out = None
        # batch normalization
        self.iter = tf.placeholder(tf.int32)
        self.best_means = {}
        self.best_variances = {}
        self.scaling_factors = {}
        self.offsets = {}
        self.update_moments = None
        self.assign_best_means = None
        self.assign_best_variances = None
        self.best_scaling_factors = {}
        self.best_offsets = {}
        self.assign_best_scaling_factors = None
        self.assign_best_offsets = None
        # others
        self.is_in_train = None

    def build_network(self, model_id, built_for_training=False):
        if built_for_training:
            if self.top_config.load_data_from_files:
                self.net_in_train = tf.placeholder(tf.float32, [None, self.top_config.input_len])
            else:
                self.net_in_train = tf.Variable(tf.zeros(shape=[self.top_config.training_mini_batch_size, self.top_config.input_len]), dtype=tf.float32)
                tf.set_random_seed(self.top_config.train_seed)
                self.assign_net_in_train = self.net_in_train.assign(tf.sqrt(tf.square(tf.random_normal([self.top_config.training_mini_batch_size, self.top_config.input_len])) +
                                                                               tf.square(tf.random_normal([self.top_config.training_mini_batch_size,
                                                                                                           self.top_config.input_len]))) / tf.sqrt(2.0))
            self.net_in_test = tf.Variable(tf.zeros(shape=[self.top_config.test_sample_num, self.top_config.input_len]), dtype=tf.float32)
            self.is_in_train = tf.placeholder(tf.bool)
            self.net_in = tf.cond(self.is_in_train, lambda: self.net_in_train, lambda: self.net_in_test)
        else:
            self.net_in = tf.placeholder(tf.float32, [None, self.top_config.input_len])

        update_ema = {}
        moving_means = {}
        moving_variances = {}
        for layer in range(self.top_config.layer_num):
            if layer == 0:
                if self.top_config.net_input_format=="H":
                    layer_input = self.net_in
                elif self.top_config.net_input_format=="H2":
                    layer_input = tf.square(self.net_in)
                shape = [self.top_config.input_len, self.top_config.node_nums[layer]]
            else:
                layer_input = layer_output
                shape = [self.top_config.node_nums[layer - 1], self.top_config.node_nums[layer]]
            weight_name = format("weight_%d_%d" % (layer, model_id))
            bias_name = format("bias_%d_%d" % (layer, model_id))
            if built_for_training:
                self.weights[layer] = tf.get_variable(name=weight_name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                self.biases[layer] = tf.get_variable(name=bias_name, shape=[shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                self.best_weights[layer] = tf.Variable(tf.ones(shape, tf.float32), dtype=tf.float32)
                self.best_biases[layer] = tf.Variable(tf.ones([shape[1]], tf.float32), dtype=tf.float32)
            else:
                self.weights[layer] = tf.Variable(tf.ones(shape, tf.float32), dtype=tf.float32, trainable=False)
                self.biases[layer] = tf.Variable(tf.ones([shape[1]], tf.float32), dtype=tf.float32, trainable=False)
            if layer != self.top_config.layer_num - 1:
                layer_output = tf.matmul(layer_input, self.weights[layer]) + self.biases[layer]
                self.offsets[layer] = tf.Variable(tf.zeros([shape[1]]), trainable=built_for_training)
                self.scaling_factors[layer] = tf.Variable(tf.ones([shape[1]]), trainable=built_for_training)
                self.best_means[layer] = tf.Variable(tf.zeros([shape[1]]), trainable=built_for_training)
                self.best_variances[layer] = tf.Variable(tf.ones([shape[1]]), trainable=built_for_training)
                self.best_offsets[layer] = tf.Variable(tf.zeros([shape[1]]), trainable=built_for_training)
                self.best_scaling_factors[layer] = tf.Variable(tf.ones([shape[1]]), trainable=built_for_training)
                if built_for_training:
                    layer_output, update_ema[layer], moving_means[layer], moving_variances[layer] = self.batchnorm(layer_output, self.offsets[layer], self.scaling_factors[layer],
                                                                                                             self.is_in_train, self.iter)
                else:
                    layer_output = tf.nn.batch_normalization(layer_output, self.best_means[layer], self.best_variances[layer], self.offsets[layer], self.scaling_factors[layer], 1e-5)
                layer_output = tf.nn.relu(layer_output)
            else:
                layer_output = tf.matmul(layer_input, self.weights[layer]) + self.biases[layer]
        self.net_out_before_sigmoid = layer_output
        self.net_out = tf.sigmoid(layer_output)

        if built_for_training:
            self.update_moments = [update_ema[i] for i in range(self.top_config.layer_num - 1)]
            self.assign_best_weights = [self.best_weights[layer].assign(self.weights[layer]) for layer in range(self.top_config.layer_num)]
            self.assign_best_biases = [self.best_biases[layer].assign(self.biases[layer]) for layer in range(self.top_config.layer_num)]
            self.assign_best_scaling_factors = [self.best_scaling_factors[layer].assign(self.scaling_factors[layer]) for layer in range(self.top_config.layer_num-1)]
            self.assign_best_offsets = [self.best_offsets[layer].assign(self.offsets[layer]) for layer in range(self.top_config.layer_num-1)]
            self.assign_best_means = [self.best_means[layer].assign(moving_means[layer]) for layer in range(self.top_config.layer_num-1)]
            self.assign_best_variances = [self.best_variances[layer].assign(moving_variances[layer]) for layer in range(self.top_config.layer_num-1)]


    def batchnorm(self, Ylogits, Offset, Scale, is_in_train, iteration):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_in_train, lambda: mean, lambda: exp_moving_avg.average(mean))
        v = tf.cond(is_in_train, lambda: variance, lambda: exp_moving_avg.average(variance))
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, Offset, Scale, bnepsilon)
        return Ybn, update_moving_averages, exp_moving_avg.average(mean), exp_moving_avg.average(variance)

    def assign_best_paras(self, sess_in):
        sess_in.run([self.assign_best_weights, self.assign_best_biases, self.assign_best_scaling_factors, self.assign_best_offsets, self.assign_best_means, self.assign_best_variances])

    def save_network_to_file(self, sess_in, model_id):
        # save network
        save_dict = {}
        for layer in range(self.top_config.layer_num):
            save_dict[format("weight_%d" % (layer))] = self.best_weights[layer]
            save_dict[format("bias_%d" % (layer))] = self.best_biases[layer]
            if layer < self.top_config.layer_num - 1:
                save_dict[format("scale_%d" % (layer))] = self.best_scaling_factors[layer]
                save_dict[format("offset_%d" % (layer))] = self.best_offsets[layer]
                save_dict[format("mean_%d" % (layer))] = self.best_means[layer]
                save_dict[format("varriance_%d" % (layer))] = self.best_variances[layer]

        model_folder = format("%s/model%d" % (self.top_config.base_folder, model_id))
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_name = format("%s/model.ckpt" % (model_folder))
        saver = tf.train.Saver(save_dict)
        saver.save(sess_in, model_name)
        print("Save the network to a file.\n")

    def restore_network(self, sess_in, model_id):
        save_dict = {}
        for layer in range(self.top_config.layer_num):
            save_dict[format("weight_%d" % (layer))] = self.weights[layer]
            save_dict[format("bias_%d" % (layer))] = self.biases[layer]
            if layer < self.top_config.layer_num - 1:
                save_dict[format("scale_%d" % (layer))] = self.scaling_factors[layer]
                save_dict[format("offset_%d" % (layer))] = self.offsets[layer]
                save_dict[format("mean_%d" % (layer))] = self.best_means[layer]
                save_dict[format("varriance_%d" % (layer))] = self.best_variances[layer]

        model_folder = format("%s/model%d" % (self.top_config.base_folder, model_id))
        model_name = format("%s/model.ckpt" % (model_folder))
        saver = tf.train.Saver(save_dict)
        saver.restore(sess_in, model_name)
        print("Restore the network from a file.\n")

    def calc_sum_rate(self, power, abs_H, noise_power, binary_enabled):
        abs_H = tf.reshape(abs_H, [-1, self.top_config.user_num, self.top_config.user_num])
        abs_H_2 = tf.square(abs_H)
        if binary_enabled:
            power = tf.round(power)
        rx_power = tf.multiply(abs_H_2, tf.reshape(power, [-1, self.top_config.user_num, 1]))
        mask = tf.eye(self.top_config.user_num)
        valid_rx_power = tf.reduce_sum(tf.multiply(rx_power, mask), axis=1)
        interference = tf.reduce_sum(tf.multiply(rx_power, 1-mask), axis=1) + noise_power
        rate = tf.log(1 + tf.divide(valid_rx_power, interference)) / tf.log(2.0)
        sum_rate = tf.reduce_mean(tf.reduce_sum(rate, axis=1))
        return sum_rate

    def get_sum_rate_vector(self, power, abs_H, noise_power, binary_enabled):
        abs_H = tf.reshape(abs_H, [-1, self.top_config.user_num, self.top_config.user_num])
        abs_H_2 = tf.square(abs_H)
        if binary_enabled:
            power = tf.round(power)
        rx_power = tf.multiply(abs_H_2, tf.reshape(power, [-1, self.top_config.user_num, 1]))
        mask = tf.eye(self.top_config.user_num)
        valid_rx_power = tf.reduce_sum(tf.multiply(rx_power, mask), axis=1)
        interference = tf.reduce_sum(tf.multiply(rx_power, 1-mask), axis=1) + noise_power
        rate = tf.log(1 + tf.divide(valid_rx_power, interference)) / tf.log(2.0)
        sum_rate = tf.reduce_sum(rate, axis=1)
        return sum_rate

    def get_per_user_rate(self, power, abs_H, noise_power, binary_enabled):  # return rates of each user for each sample in the batch
        abs_H = tf.reshape(abs_H, [-1, self.top_config.user_num, self.top_config.user_num])
        abs_H_2 = tf.square(abs_H)
        if binary_enabled:
            power = tf.round(power)
        rx_power = tf.multiply(abs_H_2, tf.reshape(power, [-1, self.top_config.user_num, 1]))
        mask = tf.eye(self.top_config.user_num)
        valid_rx_power = tf.reduce_sum(tf.multiply(rx_power, mask), axis=1)
        interference = tf.reduce_sum(tf.multiply(rx_power, 1-mask), axis=1) + noise_power
        per_user_rate = tf.log(1 + tf.divide(valid_rx_power, interference)) / tf.log(2.0)
        return per_user_rate

    def calc_min_feasible_power(self, abs_H, min_rates, noise_power):
        abs_H_2 = tf.reshape(tf.square(abs_H), [-1, self.top_config.user_num, self.top_config.user_num])
        check_mat = tf.matrix_transpose(abs_H_2)
        diag_part = tf.matrix_diag_part(abs_H_2) + 1e-10
        diag_zeros = tf.zeros(tf.shape(diag_part))
        diag_part = tf.reshape(diag_part, [-1, self.top_config.user_num, 1])
        check_mat = tf.divide(check_mat, diag_part)
        check_mat = tf.matrix_set_diag(check_mat, diag_zeros)
        min_snrs = tf.cast(tf.reshape(2**min_rates-1, [self.top_config.user_num, 1]), tf.float32)
        check_mat = tf.multiply(check_mat, min_snrs)
        u = np.divide(min_snrs, diag_part) * noise_power
        inv_id_sub_check_mat = tf.matrix_inverse(tf.subtract(tf.eye(self.top_config.user_num), check_mat))
        min_feasible_power = tf.matmul(inv_id_sub_check_mat, u)
        min_feasible_power = tf.reshape(min_feasible_power, [-1, self.top_config.user_num])
        return min_feasible_power

    def constraint_loss(self, power, abs_H, noise_power):
        abs_H = tf.reshape(abs_H, [-1, self.top_config.user_num, self.top_config.user_num])
        abs_H_2 = tf.square(abs_H)
        rx_power = tf.multiply(abs_H_2, tf.reshape(power, [-1, self.top_config.user_num, 1]))
        mask = tf.eye(self.top_config.user_num)
        valid_rx_power = tf.reduce_sum(tf.multiply(rx_power, mask), axis=1)
        interference = tf.reduce_sum(tf.multiply(rx_power, 1 - mask), axis=1) + noise_power
        # loss = tf.reduce_mean(tf.reduce_sum(tf.exp(2**self.top_config.min_rates-1 - tf.divide(valid_rx_power, interference)), axis=1))
        loss = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(2 ** (self.top_config.min_rates) - 1 - tf.divide(valid_rx_power, interference)), axis=1))
        return loss

    def train_network(self, model_id):
        data_io = DataIO(self.top_config)  # construct class for loading data

        self.build_network(model_id, True)  # build the network for training

        assign_net_in_test = self.net_in_test.assign(data_io.load_all_test_data())

        ## define loss function for training the nework
        sum_rate_dnn = self.calc_sum_rate(self.net_out, self.net_in, self.top_config.noise_power, False)
        per_user_rate = self.get_per_user_rate(self.net_out, self.net_in, self.top_config.noise_power, False)
        loss = tf.negative(sum_rate_dnn) + self.top_config.constraint_loss_scale * self.constraint_loss(self.net_out, self.net_in, self.top_config.noise_power)

        ## define optimizer for training the network
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        # init operation
        init = tf.global_variables_initializer()

        # create a session
        sess = tf.Session()
        sess.run(init)

        # load data for test
        sess.run(assign_net_in_test)

        if self.top_config.flag_restore_network:
            self.restore_network(sess, model_id)
        # calculate the loss before training and assign it to min_loss
        max_sum_rate_dnn_v, max_hit_rate = self.test_network_online(data_io, per_user_rate, sess)

        self.assign_best_paras(sess)

        # Train
        epoch = 0
        if not os.path.exists(self.top_config.base_folder):
            os.makedirs(self.top_config.base_folder)

        start = datetime.datetime.now()

        while epoch < self.top_config.epoch_num:
            epoch += 1
            if self.top_config.load_data_from_files:
                batch_net_in = data_io.gen_a_training_mini_batch(self.top_config.training_mini_batch_size)
                sess.run([optimizer], feed_dict={self.net_in_train: batch_net_in, self.is_in_train:True, self.iter:epoch})
                sess.run([self.update_moments], feed_dict={self.net_in_train: batch_net_in, self.is_in_train: True, self.iter: epoch})
            else:
                sess.run(self.assign_net_in_train)
                sess.run(optimizer, feed_dict={self.is_in_train:True})
                sess.run(self.update_moments, feed_dict={self.is_in_train:True, self.iter:epoch})
            if epoch % self.top_config.check_interval == 0 or epoch == self.top_config.epoch_num:
                sum_rate_dnn_v, hit_rate = self.test_network_online(data_io, per_user_rate, sess)
                if sum_rate_dnn_v > max_sum_rate_dnn_v:
                    max_sum_rate_dnn_v = sum_rate_dnn_v
                    max_hit_rate = hit_rate
                    self.assign_best_paras(sess)

                print(epoch, sum_rate_dnn_v, hit_rate, max_sum_rate_dnn_v, max_hit_rate)


        end = datetime.datetime.now()

        self.save_network_to_file(sess, model_id)
        sess.close()
        print('Used time for training: %ds' % (end - start).seconds)

    def test_network_online(self, data_io, per_user_rate, sess_in):
        if self.top_config.load_data_from_files:
            per_user_rate_v = sess_in.run(per_user_rate, feed_dict={self.is_in_train: False, self.net_in_train:np.zeros([1, self.top_config.input_len],
                                                                                                                                                np.float32)})
        else:
            per_user_rate_v = sess_in.run(per_user_rate, feed_dict={self.is_in_train: False})

        feasible_flag = np.sum(per_user_rate_v < self.top_config.min_rates - 1e-6, axis=1) == 0
        hit_rate = np.sum(feasible_flag) / float(self.top_config.test_sample_num)
        per_user_rate_v[~feasible_flag, :] = self.top_config.min_rates
        sum_rate = np.mean(np.sum(per_user_rate_v, axis=1))
        return sum_rate, hit_rate   # hit rate means the probability that the nework outputs a feasible solution.