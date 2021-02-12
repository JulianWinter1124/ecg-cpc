import torch

import temporal_to_image_converter as converter

if __name__ == '__main__':
    model_f = '../models/18_01_21-14/baseline_modelstate_epoch200.pt'
    model_state_dict = torch.load(model_f)['model_state_dict']
    print(model_state_dict.keys())
    for k in model_state_dict.keys():
        if 'weight' in k:
            conv = model_state_dict[k].cpu()
            print('layer weight shape:', conv.shape)
            converter.kernel_to_image(k, conv)
