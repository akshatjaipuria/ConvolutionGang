import math
import matplotlib.pyplot as plt


def calc_cycle(step_size, iteration):
    cycle = math.floor(1 + (iteration / (2 * step_size)))
    return cycle


def calc_x(step_size, iteration):
    cycle = calc_cycle(step_size, iteration)
    x = abs((iteration / step_size) - (2 * cycle) + 1)
    return x


def calc_lr_t(lr_min, lr_max, x):
    lr_t = lr_min + (lr_max - lr_min) * (1 - x)
    return lr_t


def plot_cycle(lr_min, lr_max, iterations, step_size):
    fig = plt.figure()
    for iteration in range(1, iterations + 1):
        x = calc_x(step_size, iteration)
        lr_t = calc_lr_t(lr_min, lr_max, x)
        plt.scatter(iteration, lr_t, c='black', s=0.1)
    plt.xlabel("Iterations")
    plt.ylabel("Learning Rate")
    plt.show()


# lets initialize lr_min and lr_max on own
lr_min = 0.001
lr_max = 0.1
iterations = 3000
step_size = 500

plot_cycle(lr_min, lr_max, iterations, step_size)
