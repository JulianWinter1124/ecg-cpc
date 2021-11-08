import matplotlib.pyplot as plt
import numpy as np

def plot_kernel_weights(kernel_weights):
    kw = np.repeat(kernel_weights, 1000, axis=0)
    plt.yticks([])
    plt.imshow(kw, interpolation='none', cmap=plt.cm.get_cmap('Blues'))
    plt.show()

def plot_kernel_histogram(kernel_weights):
    unique_vals, counts = np.unique(kernel_weights, return_counts=True)
    n_bins = np.arange(min(kernel_weights), max(kernel_weights))
    plt.xticks(n_bins)
    plt.xlim(min(kernel_weights)-1, max(kernel_weights)+1)
    n, bin, patches = plt.hist(kernel_weights, bins=n_bins, rwidth=0.5)
    for i in range(len(n)):
        plt.text(bin[i], n[i], str(n[i]))
    plt.show()


def plot_receptivefield_bardiagram(kernel_weights):
    plt.bar(np.arange(len(kernel_weights)), kernel_weights) #height=h, align='center'
    plt.show()

def plot_receptivefield_plot(kernel_weights, name=''):
    plt.title(name)
    plt.plot(np.arange(len(kernel_weights)), kernel_weights) #height=h, align='center'
    plt.show()

def plot_receptivefield_plot_repeated(kernel_weights, name='', repeat_shift:list=None):
    if not repeat_shift is None:
        plt.title(name)
        for i in range(len(repeat_shift)+1):
            plt.plot(np.arange(len(kernel_weights))+sum(repeat_shift[0:i]), kernel_weights) #height=h, align='center'
        plt.show()

def plot_multiple_receptivefield_plot(kernel_weights_list: list, name=''):
    plt.title(name)
    for kw in kernel_weights_list:
        plt.plot(np.arange(len(kw)), kw) #height=h, align='center'
    plt.show()