
import tensorflow as tf
import numpy as np
import datetime
import os
import ComLib as clb


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
            if self.top_config.csi_distribution == "Rayleigh":
                CH = 1 / np.sqrt(2) * (rng.randn(mini_batch_size, K ** 2) + 1j * rng.randn(mini_batch_size, K ** 2))
                batch_net_in = np.abs(CH)
            elif self.top_config.csi_distribution == "Rice":
                CH = 1 / 2 * (rng.randn(mini_batch_size, K ** 2) + 1 + 1j * (1+rng.randn(mini_batch_size, K ** 2)))
                batch_net_in = np.abs(CH)
            elif self.top_config.csi_distribution == "Geometry":
                batch_net_in = clb.generate_geometry_CSI(K, mini_batch_size, rng)
            else:
                print("Invalid CSI distribution!")
                exit(0)
        return batch_net_in


    def gen_a_training_mini_batch(self, mini_batch_size):
        batch_net_in = self.gen_a_mini_batch(mini_batch_size, self.rng_training, "Training")

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

    def build_fcn(self, model_id, built_for_training=False):
        self.generate_net_in(built_for_training)

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
                if self.top_config.initialization=="Xavier":
                    self.weights[layer] = tf.get_variable(name=weight_name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                    self.biases[layer] = tf.get_variable(name=bias_name, shape=[shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                elif self.top_config.initialization=="TN":
                    self.weights[layer] = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.35, dtype=np.float32), dtype=tf.float32)
                    self.biases[layer] = tf.Variable(tf.zeros(shape=[shape[1]], dtype=tf.float32), dtype=tf.float32)
                self.best_weights[layer] = tf.Variable(tf.ones(shape, tf.float32), dtype=tf.float32)
                self.best_biases[layer] = tf.Variable(tf.ones([shape[1]], tf.float32), dtype=tf.float32)
            else:
                self.weights[layer] = tf.Variable(tf.ones(shape, tf.float32), dtype=tf.float32, trainable=False)
                self.biases[layer] = tf.Variable(tf.ones([shape[1]], tf.float32), dtype=tf.float32, trainable=False)

            layer_output = tf.matmul(layer_input, self.weights[layer]) + self.biases[layer]
            self.offsets[layer] = tf.Variable(tf.zeros([shape[1]]), trainable=built_for_training)
            self.scaling_factors[layer] = tf.Variable(tf.ones([shape[1]]), trainable=built_for_training)
            self.best_means[layer] = tf.Variable(tf.zeros([shape[1]]), trainable=built_for_training)
            self.best_variances[layer] = tf.Variable(tf.ones([shape[1]]), trainable=built_for_training)
            self.best_offsets[layer] = tf.Variable(tf.zeros([shape[1]]), trainable=built_for_training)
            self.best_scaling_factors[layer] = tf.Variable(tf.ones([shape[1]]), trainable=built_for_training)
            if layer != self.top_config.layer_num - 1:
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
            self.generate_assign_ops(moving_means, moving_variances, self.top_config.layer_num)


    def build_network(self, model_id, built_for_training=False):
        self.build_fcn(model_id, built_for_training)

    def batchnorm(self, Ylogits, Offset, Scale, is_in_train, iteration, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_in_train, lambda: mean, lambda: exp_moving_avg.average(mean))
        v = tf.cond(is_in_train, lambda: variance, lambda: exp_moving_avg.average(variance))
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, Offset, Scale, bnepsilon)
        return Ybn, update_moving_averages, exp_moving_avg.average(mean), exp_moving_avg.average(variance)

    def generate_geometric_csi(self):
        area_length = 10
        alpha = 2
        batch_size = self.top_config.training_mini_batch_size
        user_num = self.top_config.user_num
        tx_pos_x = area_length * tf.random_uniform([batch_size, user_num], 0, 1)
        tx_pos_y = area_length * tf.random_uniform([batch_size, user_num], 0, 1)
        rx_pos_x = area_length * tf.random_uniform([batch_size, user_num], 0, 1)
        rx_pos_y = area_length * tf.random_uniform([batch_size, user_num], 0, 1)

        tx_pos_x = tf.reshape(tx_pos_x, [batch_size, user_num, 1]) + np.zeros([1, 1, user_num])
        tx_pos_y = tf.reshape(tx_pos_y, [batch_size, user_num, 1]) + np.zeros([1, 1, user_num])
        rx_pos_x = tf.reshape(rx_pos_x, [batch_size, 1, user_num]) + np.zeros([1, user_num, 1])
        rx_pos_y = tf.reshape(rx_pos_y, [batch_size, 1, user_num]) + np.zeros([1, user_num, 1])
        d = tf.sqrt(tf.square(tx_pos_x - rx_pos_x) + tf.square(tx_pos_y - rx_pos_y))
        G = tf.divide(1, 1 + d ** alpha)
        rayleigh_coeff = (tf.square(tf.random_normal([batch_size, user_num, user_num])) + tf.square(tf.random_normal([batch_size, user_num, user_num]))) / 2
        random_csi = tf.sqrt(tf.reshape(G * rayleigh_coeff, [batch_size, user_num**2]))
        return random_csi

    def generate_net_in(self, built_for_training):
        if built_for_training:
            if self.top_config.load_data_from_files:
                self.net_in_train = tf.placeholder(tf.float32, [None, self.top_config.input_len])
            else:
                self.net_in_train = tf.Variable(tf.zeros(shape=[self.top_config.training_mini_batch_size, self.top_config.input_len]), dtype=tf.float32)
                tf.set_random_seed(self.top_config.train_seed)
                if self.top_config.csi_distribution=="Rice":
                    random_gen_csi = tf.sqrt(tf.square(1+tf.random_normal([self.top_config.training_mini_batch_size, self.top_config.input_len])) +
                                             tf.square(1+tf.random_normal([self.top_config.training_mini_batch_size,
                                                                         self.top_config.input_len]))) / 2
                elif self.top_config.csi_distribution=="Rayleigh":
                    random_gen_csi = tf.sqrt(tf.square(tf.random_normal([self.top_config.training_mini_batch_size, self.top_config.input_len])) +
                                             tf.square(tf.random_normal([self.top_config.training_mini_batch_size,
                                                                         self.top_config.input_len]))) / tf.sqrt(2.0)
                elif self.top_config.csi_distribution=="Geometry":
                    random_gen_csi = self.generate_geometric_csi()

                self.assign_net_in_train = self.net_in_train.assign(random_gen_csi)

            self.net_in_test = tf.Variable(tf.zeros(shape=[self.top_config.test_sample_num, self.top_config.input_len]), dtype=tf.float32)
            self.is_in_train = tf.placeholder(tf.bool)
            self.net_in = tf.cond(self.is_in_train, lambda: self.net_in_train, lambda: self.net_in_test)
        else:
            self.net_in = tf.placeholder(tf.float32, [None, self.top_config.input_len])

    def generate_assign_ops(self, moving_means, moving_variances, total_layer_num):
        self.assign_best_weights = [self.best_weights[layer].assign(self.weights[layer]) for layer in range(total_layer_num)]
        self.assign_best_biases = [self.best_biases[layer].assign(self.biases[layer]) for layer in range(total_layer_num)]
        self.assign_best_scaling_factors = [self.best_scaling_factors[layer].assign(self.scaling_factors[layer]) for layer in range(total_layer_num - 1)]
        self.assign_best_offsets = [self.best_offsets[layer].assign(self.offsets[layer]) for layer in range(total_layer_num - 1)]
        self.assign_best_means = [self.best_means[layer].assign(moving_means[layer]) for layer in range(total_layer_num - 1)]
        self.assign_best_variances = [self.best_variances[layer].assign(moving_variances[layer]) for layer in range(total_layer_num - 1)]

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

    def get_sum_rate_vector(self, power, abs_H, noise_power, binary_enabled):  # return a vector of sum rate for each sample in the batch
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

    def train_network(self, model_id):
        data_io = DataIO(self.top_config)  # construct class for loading data

        self.build_network(model_id, True)  # build the network for training

        assign_net_in_test = self.net_in_test.assign(data_io.load_all_test_data())

        ## define loss function for training the nework
        sum_rate_dnn = self.calc_sum_rate(self.net_out, self.net_in, self.top_config.noise_power, False)
        sum_rate_dnn_binary = self.calc_sum_rate(self.net_out, self.net_in, self.top_config.noise_power, True)
        loss = tf.negative(sum_rate_dnn)

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

        if self.top_config.load_data_from_files==False:
            print("Note: top_config.load_data_from_files is False. The training data is generated online during training. This is done in tensorflow and "
                  "data_io is not used.")
        # calculate the loss before training and assign it to min_loss
        max_sum_rate_dnn_v, max_sum_rate_dnn_v_binary = self.test_network_online(data_io, sum_rate_dnn, sum_rate_dnn_binary, sess)
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
                sum_rate_dnn_v, sum_rate_dnn_v_binary = self.test_network_online(data_io, sum_rate_dnn, sum_rate_dnn_binary, sess)
                if sum_rate_dnn_v > max_sum_rate_dnn_v:
                    max_sum_rate_dnn_v = sum_rate_dnn_v
                if sum_rate_dnn_v_binary > max_sum_rate_dnn_v_binary:
                    max_sum_rate_dnn_v_binary = sum_rate_dnn_v_binary
                    self.assign_best_paras(sess)
                if epoch % 50 == 0:
                    print(epoch, sum_rate_dnn_v, sum_rate_dnn_v_binary, max_sum_rate_dnn_v, max_sum_rate_dnn_v_binary)

        end = datetime.datetime.now()

        self.save_network_to_file(sess, model_id)
        sess.close()
        print('Used time for training: %ds' % (end - start).seconds)

    def test_network_online(self, data_io, sum_rate_dnn, sum_rate_dnn_binary, sess_in):

        if self.top_config.load_data_from_files:
            rate, rate_binary = sess_in.run([sum_rate_dnn, sum_rate_dnn_binary], feed_dict={self.is_in_train: False, self.net_in_train:np.zeros([1,
                                                                                                                                                 self.top_config.input_len], np.float32)})
        else:
            rate, rate_binary = sess_in.run([sum_rate_dnn, sum_rate_dnn_binary], feed_dict={self.is_in_train: False})
        return rate, rate_binary
