import numpy as np


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
        print('In', output_length)
        output_length = _calc_conv1d_output_length(output_length, padding, dilation, kernel_size, stride)
        print('out', output_length)
    return output_length

def _calc_conv1d_output_length(input_length, padding, dilation, kernel_size, stride):
    return int((input_length + 2*padding - dilation*(kernel_size-1)-1)/stride+1)
