import numpy as np
import torch
from torch.nn import functional as F

import util.layer_visualization as lv


def calc_conv1ds_output_length(input_length, kernel_sizes:list, paddings:list=None, dilations:list=None, strides:list=None):
    assert (paddings is None or len(paddings) == len(kernel_sizes)) and (dilations is None or len(dilations) == len(kernel_sizes)) and (strides is None or len(strides) == len(kernel_sizes))
    if paddings is None:
        paddings = [0]*len(kernel_sizes)
    if dilations is None:
        dilations = [1]*len(kernel_sizes)
    if strides is None:
        strides = [1]*len(kernel_sizes)
    output_length = input_length
    for padding, dilation, kernel_size, stride in zip(paddings, dilations, kernel_sizes, strides):
        print('In', output_length, end='->')
        output_length = _calc_conv1d_output_length(output_length, padding, dilation, kernel_size, stride)
        print('out', output_length)
    return output_length

def _calc_conv1d_output_length(input_length, padding, dilation, kernel_size, stride):
    return int((input_length + 2*padding - dilation*(kernel_size-1)-1)/stride+1)


def calc_conv1ds_input_length(output_length, kernel_sizes:list, paddings:list=None, dilations:list=None, strides:list=None):
    assert (paddings is None or len(paddings) == len(kernel_sizes)) and (
                dilations is None or len(dilations) == len(kernel_sizes)) and (
                       strides is None or len(strides) == len(kernel_sizes))
    if paddings is None:
        paddings = [0] * len(kernel_sizes)
    if dilations is None:
        dilations = [1] * len(kernel_sizes)
    if strides is None:
        strides = [1] * len(kernel_sizes)
    input_length = output_length
    for padding, dilation, kernel_size, stride in reversed(list(zip(paddings, dilations, kernel_sizes, strides))):
        print('In', input_length, end='->')
        input_length = _calc_conv1d_input_length(input_length, padding, dilation, kernel_size, stride)
        print('out', input_length)
    return input_length

def calc_conv1ds_input_length_range(output_length, kernel_sizes:list, paddings:list=None, dilations:list=None, strides:list=None):
    assert (paddings is None or len(paddings) == len(kernel_sizes)) and (
                dilations is None or len(dilations) == len(kernel_sizes)) and (
                       strides is None or len(strides) == len(kernel_sizes))
    if paddings is None:
        paddings = [0] * len(kernel_sizes)
    if dilations is None:
        dilations = [1] * len(kernel_sizes)
    if strides is None:
        strides = [1] * len(kernel_sizes)
    in_min = in_max = output_length
    for padding, dilation, kernel_size, stride in reversed(list(zip(paddings, dilations, kernel_sizes, strides))):
        print('In', in_min, in_max, end='->')
        in_min = _calc_conv1d_input_length(in_min, padding, dilation, kernel_size, stride)
        in_max = _calc_conv1d_input_length(in_max+((stride-1)/stride), padding, dilation, kernel_size, stride) #assumes maximum possible error
        print('out', in_min, in_max)
    return in_min, in_max

def _calc_conv1d_input_length(output_length, padding, dilation, kernel_size, stride):
    return int((output_length-1)*stride - 2*padding + dilation*(kernel_size-1) + 1)

def calc_conv1d_input_receptive_field(output_length, kernel_sizes: list, paddings: list = None, dilations: list = None, strides: list = None, weights='balanced'):
    assert (paddings is None or len(paddings) == len(kernel_sizes)) and (
            dilations is None or len(dilations) == len(kernel_sizes)) and (
                   strides is None or len(strides) == len(kernel_sizes))
    if paddings is None:
        paddings = [0] * len(kernel_sizes)
    if dilations is None:
        dilations = [1] * len(kernel_sizes)
    if strides is None:
        strides = [1] * len(kernel_sizes)

    receptive_out = torch.ones((1, 1, output_length))
    for padding, dilation, kernel_size, stride in reversed(list(zip(paddings, dilations, kernel_sizes, strides))):
        #print('In:', receptive_out, end='->')
        receptive_out = _calc_conv1d_input_receptive_field(receptive_out=receptive_out, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, kernel_weights=weights)
        #print('out:', receptive_out)
    return receptive_out.squeeze().numpy()

def _calc_conv1d_input_receptive_field(receptive_out, kernel_size, padding, dilation, stride, kernel_weights='balanced'):
    if kernel_weights is None or kernel_weights == 'balanced': #Good to see how many each value is used in the final output
        kernel_weights = torch.ones(1, 1, kernel_size)
    elif kernel_weights == 'left': #Good to see start of new value in outputmap
        kernel_weights = torch.zeros(1, 1, kernel_size)
        kernel_weights[0, 0, 0] = 1
    elif kernel_weights == 'right': #Good to see end of new value in outputmap
        kernel_weights = torch.zeros(1, 1, kernel_size)
        kernel_weights[0, 0, -1] = 1
    receptive_input_map = F.conv_transpose1d(input=receptive_out, weight=kernel_weights, stride=stride, dilation=dilation, padding=padding)
    return receptive_input_map



if __name__ == '__main__':
    #To quickly run calculations:
    # print(calc_conv1ds_output_length(512, kernel_sizes=[3,3,3,3,3,3], dilations=[1,3,9,27,27*3,27*3*3], paddings=[3,3,3,3,3,3]))
    # print(calc_conv1ds_output_length(465, kernel_sizes=[10, 8, 4, 4, 4], strides = [5, 4, 2, 2, 2]))
    # print(calc_conv1ds_input_length(1, kernel_sizes=[10, 8, 4, 4, 4], strides=[5, 4, 2, 2, 2]))
    # print(calc_conv1ds_input_length(1, kernel_sizes = [8, 6, 3, 3, 3], strides = [4, 2, 1, 1, 1], dilations = [1, 1, 1, 3, 9]))
    # print(calc_conv1ds_input_length(1,kernel_sizes = [7, 3, 3, 3, 3], strides = [2, 1, 1, 1, 1], dilations = [1, 1, 3, 9, 27]))
    # print(calc_conv1ds_output_length(9500, kernel_sizes=[7, 3, 3, 3, 3], strides=[2, 1, 1, 1, 1], dilations=[1, 1, 3, 9, 27]))
    print(calc_conv1ds_output_length(9500, kernel_sizes=[10, 8, 4, 4, 4], strides=[5, 4, 2, 2, 2]))
    print(calc_conv1ds_input_length(1, kernel_sizes=[10, 8, 4, 4, 4], strides=[5, 4, 2, 2, 2]))
    print(calc_conv1ds_input_length_range(57, kernel_sizes=[10, 8, 4, 4, 4], strides=[5, 4, 2, 2, 2]))
    #print(calc_conv1d_input_receptive_field(1, kernel_sizes=[10, 8, 4, 4, 4], strides=[5, 4, 2, 2, 2]))
    rf = calc_conv1d_input_receptive_field(57, kernel_sizes=[10, 8, 4, 4, 4], strides=[5, 4, 2, 2, 2], weights='balanced')
    print(np.count_nonzero(rf))
    lv.plot_receptivefield_plot(rf)
    rfr = calc_conv1d_input_receptive_field(57, kernel_sizes=[10, 8, 4, 4, 4], strides=[5, 4, 2, 2, 2], weights='right')
    rfl = calc_conv1d_input_receptive_field(57, kernel_sizes=[10, 8, 4, 4, 4], strides=[5, 4, 2, 2, 2], weights='left')
    lv.plot_multiple_receptivefield_plot(rfr, rfl)
