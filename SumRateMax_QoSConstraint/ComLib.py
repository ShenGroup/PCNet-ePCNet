##
import numpy as np

def generate_CSI(K, num_H, rng):
    X = np.zeros((num_H, K**2))
    for loop in range(num_H):
        CH = 1/np.sqrt(2)*(rng.randn(1,K**2)+1j*rng.randn(1,K**2))
        X[loop, :]=abs(CH)
    return X

def calc_sum_rate(power, abs_H, noise_power, binary_enabled):  # return the rate of each net input
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

def calc_per_user_rate(power, abs_H, noise_power, binary_enabled):  # return the rate for each user
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
    return rate


def normalize_power(power, Pmax):  # normalize the power solutions to make its maximum power is Pmax
    max_power = np.max(power, axis=1, keepdims=True)
    scale_factor = np.divide(Pmax, max_power)
    power = np.multiply(scale_factor, power)
    return power

def check_feasibility(batch_absH_2, min_snrs, noise_power): # absH_2 means |H|^2
    user_num = np.size(min_snrs)
    batch_size = np.size(batch_absH_2, 0)
    check_mat = np.zeros([batch_size, user_num, user_num])
    for i in range(user_num):
        check_mat[:, i, :] = min_snrs[i]*batch_absH_2[:, :, i] / batch_absH_2[:, i:i+1, i]
        check_mat[:, i, i] = 0

    eig_value, _ = np.linalg.eig(check_mat)
    eig_value *= (np.imag(eig_value) == 0)
    eig_value = np.real(eig_value)
    max_eig = np.max(eig_value, axis=1)

    feasible_flag = np.zeros(batch_size, dtype=np.bool)
    min_used_power = np.zeros([batch_size, user_num])
    for i in range(batch_size):
        if max_eig[i] > 1:
            continue

        absH_2 = batch_absH_2[i, :, :]
        u = np.divide(min_snrs, np.diag(absH_2)) * noise_power
        u = np.reshape(u, [user_num, 1])
        P = np.matmul(np.linalg.inv(np.identity(user_num) - check_mat[i,:,:]), u)
        P = np.reshape(P, [user_num])
        if np.sum(P<0)>0 or np.sum(P>1)>0:
            continue

        feasible_flag[i] = True
        min_used_power[i, :] = P
    return feasible_flag, min_used_power