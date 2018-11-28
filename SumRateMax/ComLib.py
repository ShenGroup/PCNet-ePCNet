
import numpy as np


def generate_geometry_CSI(user_num, batch_size, rng):  # generate a batch of structural data in RoundRobin paper.
    area_length = 10
    alpha = 2
    tx_pos = np.zeros([batch_size, user_num, 2])
    rx_pos = np.zeros([batch_size, user_num, 2])
    rayleigh_coeff = np.zeros([batch_size, user_num, user_num])
    for i in range(batch_size):
        tx_pos[i, :, :] = rng.rand(user_num, 2) * area_length
        rx_pos[i, :, :] = rng.rand(user_num, 2) * area_length
        rayleigh_coeff[i, :, :] = (np.square(rng.randn(user_num, user_num)) + np.square(rng.randn(user_num, user_num))) / 2

    tx_pos_x = np.reshape(tx_pos[:, :, 0], [batch_size, user_num, 1]) + np.zeros([1, 1, user_num])
    tx_pos_y = np.reshape(tx_pos[:, :, 1], [batch_size, user_num, 1]) + np.zeros([1, 1, user_num])
    rx_pos_x = np.reshape(rx_pos[:, :, 0], [batch_size, 1, user_num]) + np.zeros([1, user_num, 1])
    rx_pos_y = np.reshape(rx_pos[:, :, 1], [batch_size, 1, user_num]) + np.zeros([1, user_num, 1])
    d = np.sqrt(np.square(tx_pos_x - rx_pos_x) + np.square(tx_pos_y - rx_pos_y))
    G = np.divide(1, 1 + d**alpha)
    G = G * rayleigh_coeff
    return np.sqrt(np.reshape(G, [batch_size, user_num ** 2]))


def generate_rayleigh_CSI(K, num_H, rng):
    X = np.zeros((num_H, K ** 2))
    for loop in range(num_H):
        #  generate num_H samples of CSI one by one instead of generating all together in order to generate the same samples with different num_H and the same rng.
        CH = 1 / np.sqrt(2) * (rng.randn(1, K ** 2) + 1j * rng.randn(1, K ** 2))
        X[loop, :] = abs(CH)
    return X


def generate_rice_CSI(K, num_H, rng):
    X = np.zeros((num_H, K ** 2))
    for loop in range(num_H):
        #  generate num_H samples of CSI one by one instead of generating all together in order to generate the same samples with different num_H and the same rng.
        CH = 1 / 2 * (1 + rng.randn(1, K ** 2) + 1j * (1 + rng.randn(1, K ** 2)))
        X[loop, :] = abs(CH)
    return X


def generate_CSI(K, num_H, rng, distribution):
    if distribution=="Rayleigh":
        abs_H = generate_rayleigh_CSI(K, num_H, rng)
    elif distribution=="Rice":
        abs_H = generate_rice_CSI(K, num_H, rng)
    elif distribution=="Geometry":
        abs_H = generate_geometry_CSI(K, num_H, rng)
    else:
        print("Invalid CSI distribution.")
        exit(0)

    return abs_H


def normalize_power(power, Pmax):  ## this function normalizes power profile to let its maximum element equal to Pmax
    if np.size(np.shape(power)) == 1:
        max_power = np.max(power)
        power = np.multiply(power, Pmax / max_power)
    else:
        max_power = np.max(power, axis=1, keepdims=True)
        power = np.multiply(power, np.divide(Pmax, max_power))

    return power


def calc_sum_rate(power, abs_H, noise_power, binary_enabled):
    if np.size(np.shape(power)) == 1:
        power = np.reshape(power, [1, -1])
    user_num = np.size(power, axis=1)
    abs_H = np.reshape(abs_H, [-1, user_num, user_num])
    abs_H_2 = np.square(abs_H)
    if binary_enabled:
        power = (np.sign(power - 0.5) + 1) / 2
        power = power.astype(np.float32)
    rx_power = np.multiply(abs_H_2, np.reshape(power, [-1, user_num, 1]))
    mask = np.eye(user_num)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=1)
    interference = np.sum(np.multiply(rx_power, 1 - mask), axis=1) + noise_power
    rate = np.log(1 + np.divide(valid_rx_power, interference)) / np.log(2.0)
    sum_rate = np.sum(rate, axis=1)
    return sum_rate

def search_opt_power_ctrl(absH, Pmax, noise_power, ptrl_sln_set, n_levels):
    absH = np.multiply(absH, np.ones([np.size(ptrl_sln_set, axis=0), 1]))
    ptrl_sln_set = ptrl_sln_set * Pmax / float(n_levels)
    sum_rate = calc_sum_rate(ptrl_sln_set, absH, noise_power, False)
    idx = np.argmax(sum_rate)
    return sum_rate[idx], ptrl_sln_set[idx,:]