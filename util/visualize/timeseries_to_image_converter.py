import io
from typing import Optional, Dict, Any, List

import av
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.io import write_video
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize


# from torch.nn.functional import relu
from util.data import ecg_datasets3


class VideoWriter():
    def __init__(self, filename: str, fps: float, video_codec: str = "libx264",
                 options: Optional[Dict[str, Any]] = None):
        self.filename = filename
        self.fps = fps
        if isinstance(fps, float):
            self.fps = np.round(fps)
        self.video_codec = video_codec
        self.options = options
        self.is_open = False

    def tensor_to_video_continuous(self, video_array: torch.Tensor, convert_timeseries=False) -> None:
        """
        https://pytorch.org/docs/stable/_modules/torchvision/io.html#VideoReader
        """
        video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()
        if convert_timeseries:
            video_array = timeseries_to_image(video_array)
        if not self.is_open:
            self._container = av.open(self.filename, mode="w")
            self._stream = self._container.add_stream(self.video_codec, rate=self.fps)
            self._stream.width = video_array.shape[2]
            self._stream.height = video_array.shape[1]
            self._stream.pix_fmt = "yuv420p" if self.video_codec != "libx264rgb" else "rgb24"
            self._stream.options = self.options or {}
            self.is_open = True

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pict_type = "NONE"
            for packet in self._stream.encode(frame):
                self._container.mux(packet)

    def close(self):
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.flush()
        self._container.close()
        self.is_open = False


def tensor_to_video(filename: str,
                    video_array: torch.Tensor,
                    fps: float,
                    video_codec: str = "libx264",
                    options: Optional[Dict[str, Any]] = None):
    return write_video(filename, video_array, fps, video_codec, options)


def timeseries_to_image(data: torch.Tensor, title='ECG-data visualization',
                        ground_truth: list = None, filename: str = None, show=False, save=True):
    if len(data.shape) == 2:
        data = data[np.newaxis]
    batches, width, height = data.shape
    if not filename:
        filename = title.replace(' ', '')+'.png'
    for batch in range(batches):
        fig, axs = plt.subplots(data.shape[-1], 1, figsize=(30,20))
        plt.xlim((0, width))
        plt.ylim((0, 1))

        if not ground_truth is None:
            title += "| Correct classes: " + ", ".join(map(str, np.nonzero(ground_truth[batch].numpy())[0])) + ". "
        fig.suptitle(title, fontsize=24)
        fig.tight_layout()
        for i, ax in enumerate(axs):
            ax.set_xlim((0, width))
            ax.axis('off')
            d = data[batch, :, i]
            ax.set_ylim((d.min()-0.1, d.max()+0.1))
            ax.plot(range(width), d, color='red')
        ax.axis('on')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        axs[-1].get_yaxis().set_visible(False)
        plt.xlabel('Time (500 steps = 1 second)', fontsize=18)

        if save:
            plt.savefig(f"{filename.split('.')[-2]}{batch}.{filename.split('.')[-1]}", dpi=fig.dpi)
        if show:
            plt.show()
        plt.close()
        print('Finished')


def timeseries_to_image_with_gradient(data: torch.Tensor, labels: torch.Tensor, grad: torch.Tensor,
                                      pred: torch.Tensor = None, model_tresholds=None, grad_alteration='none',
                                      title=None, class_name=None, filenames: list = None, show=False, save=True):
    batches, width, height = data.shape
    for batch in range(batches):
        cmap_green = sns.light_palette("seagreen", as_cmap=True)
        fig, axs = plt.subplots(data.shape[-1], 1, figsize=(30, 20))
        plt.xlim((0, width))
        plt.ylim((0, 1))
        if grad_alteration is None or grad_alteration == 'none':
            gradient = grad[batch]
        elif grad_alteration == 'abs':
            gradient = grad[batch].abs()
        elif grad_alteration == 'relu':
            gradient = torch.nn.functional.relu(grad[batch]) ** 2  # all values >= 0
        grad_norm = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        title = title or f'Gradient visualization for class:{class_name} as label\n'

        if not labels is None:
            title += "Correct classes: " + ", ".join(map(str, np.nonzero(labels[batch].numpy())[0])) + ". "
        if not (pred is None or model_tresholds is None):
            binary_pred = pred[batch] >= model_tresholds
            title += "Predicted classes: " + ", ".join(map(str, np.nonzero(binary_pred.numpy())[0])) + ". "
        fig.suptitle(title)
        fig.tight_layout()

        for i, ax in enumerate(axs):
            ax.set_xlim((0, width))
            ax.set_ylim((-0.1, 1.1))
            ax.axis('off')
            d = data[batch, :, i]
            g = grad_norm[:, i:i + 1].T
            ax.plot(range(width), d, color='red')
            ax.autoscale(False)
            ax.imshow(g, extent=[0, width, -0.1, 1.1], aspect='auto', cmap=cmap_green)
        if save:
            plt.savefig(
                filenames[batch] + '-class:' + str(class_name) + '-alt:' + grad_alteration + '-gradient-vis.png',
                dpi=fig.dpi)
        if show:
            plt.show()
        plt.close()


def timeseries_to_image_with_gradient_joined(data: torch.Tensor, labels: torch.Tensor, grad_list: List[torch.Tensor],
                                             pred: torch.Tensor = None, model_tresholds=None, grad_alteration='none',
                                             cutoff=0.1, title=None, class_name_list=None, filenames: list = None,
                                             show=False, save=True):
    batches, width, height = data.shape
    n_preds = len(grad_list)
    base_colors = sns.color_palette("hls", n_preds)
    cmaps = [sns.light_palette(c, as_cmap=True) for c in base_colors]
    for i in range(len(cmaps)):
        cmaps[i].set_gamma(1.)
        cmaps[i]._lut[0, :] = 0.  # Set all initial values to transparent
    for batch in range(batches):
        fig, axs = plt.subplots(data.shape[-1], 1, figsize=(30, 20))
        plt.xlim((0, width))
        plt.ylim((0, 1))
        title = title or f'Gradient visualization for class:{class_name_list} as label\n'
        title += f"Gradient Alteration: '{grad_alteration}'\n"
        if not labels is None:
            title += "Correct classes: " + ", ".join(map(str, np.nonzero(labels[batch].numpy())[0])) + ". "
        if not (pred is None or model_tresholds is None):
            binary_pred = pred[batch] >= model_tresholds
            title += "Predicted classes: " + ", ".join(map(str, np.nonzero(binary_pred.numpy())[0])) + ". "
        fig.suptitle(title)
        fig.tight_layout()
        for i, ax in enumerate(axs):
            ax.set_xlim((0, width))
            ax.set_ylim((-0.1, 1.1))
            ax.axis('off')
            d = data[batch, :, i]
            ax.plot(range(width), d, color='red')
        legend_handles = []
        grad_norms = []
        gradients = []
        color_values = []
        for n in range(n_preds):
            if grad_alteration == 'abs':
                gradient = grad_list[n][batch].abs()
            elif grad_alteration == 'relu':
                gradient = torch.nn.functional.relu(grad_list[n][batch])  # all values >= 0
            elif grad_alteration == 'abs_neg':
                gradient = grad_list[n][batch].abs()
            elif grad_alteration == 'relu_neg':
                gradient = torch.nn.functional.relu(-grad_list[n][batch])  # all values >= 0
            else:
                gradient = grad_list[n][batch]
            gradients.append(gradient.numpy())
            grad_norm = (gradient - gradient.min()) / (gradient.max() - gradient.min()).numpy()
            grad_norm[grad_norm < cutoff] = 0.
            grad_norms.append(grad_norm)
            color_values.append(cmaps[n](grad_norm))
            legend_handles.append(mpatches.Patch(color=base_colors[n], label=f"Class: {class_name_list[n]}"))
            ####NEXT IDEA: use custom color map (choose depending on argmax over n_preds)
        grad_norms = np.stack(grad_norms)
        gradients = np.stack(gradients)
        color_values = np.stack(color_values)
        # ix = np.argmax(grad_norms, axis=0)
        ix = np.argmax(gradients, axis=0)

        print(grad_norms.shape, ix.shape, color_values.shape)
        for i, ax in enumerate(axs):
            g = color_values[ix[:, i], np.arange(width), i][np.newaxis, :]  # grad_norm[:, i:i+1].T
            ax.autoscale(False)
            ax.imshow(g, extent=[0, width, -0.1, 1.1], aspect='auto', alpha=1, interpolation='none')
        fig.legend(handles=legend_handles)
        if save:
            plt.savefig(
                filenames[batch] + '-class:' + str(class_name_list) + '-alt:' + grad_alteration + '-gradient-vis.png',
                dpi=fig.dpi)
        if show:
            plt.show()
        plt.close()

def timeseries_to_image_with_gradient_cam(data: torch.Tensor, labels: torch.Tensor, grad_list: List[torch.Tensor],
                                             pred: torch.Tensor = None, model_tresholds=None, cut_off=0.4, title=None, class_name_list=None, filenames: list = None,
                                             show=False, save=True):
    batches, width, height = data.shape
    n_preds = len(grad_list)
    base_colors = sns.color_palette("hls", n_preds)
    cmaps = [sns.light_palette(c, as_cmap=True) for c in base_colors]
    for batch in range(batches):
        fig, axs = plt.subplots(data.shape[-1], 1, figsize=(30, 20))
        plt.xlim((0, width))
        plt.ylim((0, 1))
        title = title or f'Gradient visualization for class:{class_name_list} as label\n'
        if not labels is None:
            title += "Correct classes: " + ", ".join(map(str, np.nonzero(labels[batch].numpy())[0])) + ". "
        if not (pred is None or model_tresholds is None):
            binary_pred = pred[batch] >= model_tresholds
            title += "Predicted classes: " + ", ".join(map(str, np.nonzero(binary_pred.numpy())[0])) + ". "
        fig.suptitle(title)
        fig.tight_layout()
        for i, ax in enumerate(axs):
            ax.set_xlim((0, width))
            ax.axis('off')
            d = data[batch, :, i]
            ax.set_ylim((d.min()-0.1, d.max()+0.1))
            ax.plot(range(width), d, color='red')
        legend_handles = []
        for n in range(n_preds):
            legend_handles.append(mpatches.Patch(color=base_colors[n], label=f"Class: {class_name_list[n]}"))
            for i, ax in enumerate(axs):
                ax.autoscale(False)
                bottom, top = ax.get_ylim()
                r = (top-bottom)/n_preds
                if len(grad_list[n][batch].shape) == 1:
                    g = grad_list[n][batch][np.newaxis, :]
                    ax.imshow(g, extent=[0, width, top-r*(n+1), top-r*n], cmap=cmaps[n], aspect='auto', alpha=0.8, interpolation='bilinear')
                elif len(grad_list[n][batch].shape) == 2:
                    g = grad_list[n][batch][:, i:i+1].T
                    g[g<cut_off]=0
                    ax.imshow(g, extent=[0, width, top-r*(n+1), top-r*n], cmap=cmaps[n], norm=Normalize(vmin=0, vmax=1), aspect='auto', alpha=1, interpolation='bilinear')
                else:
                    print("Wrong shape for gradient:", grad_list[n][batch].shape)
        fig.legend(handles=legend_handles)
        if save:
            plt.savefig(
                filenames[batch] + '-class:' + str(class_name_list) + '-gradient-cam.png',
                dpi=fig.dpi)
        if show:
            plt.show()
        plt.close()
        print('Finished')


def kernel_to_image(layer_name, layer_weights: torch.Tensor):
    out_channels, in_channels, kernel_size = layer_weights.shape

    fig, axs = plt.subplots(1, in_channels, figsize=(out_channels // 2, in_channels // 2))
    fig.suptitle(layer_name)
    mins = layer_weights.min(dim=2)[0].unsqueeze(2)
    maxs = layer_weights.max(dim=2)[0].unsqueeze(2)
    layer_weights = (layer_weights - mins) / (maxs - mins)
    for index, ax in enumerate(axs):
        ax.imshow(layer_weights[:, index, :], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


if __name__ == '__main__':  # Usage example
    # model_f = '../models/18_01_21-14/baseline_modelstate_epoch200.pt'
    # model_state_dict = torch.load(model_f)['model_state_dict']
    # print(model_state_dict.keys())
    # for k in model_state_dict.keys():
    #     if 'weight' in k:
    #         conv = model_state_dict[k].cpu()
    #         print('layer weight shape:', conv.shape)
    #         kernel_to_image(k, conv)

    path_without_ext = '/media/julian/data/data/ECG/ptbxl_challenge/HR14099'
    d = ecg_datasets3.ECGChallengeDatasetBaseline(None, None)
    data = d._read_recording_file(path_without_ext)
    labels = d._read_header_file(path_without_ext)
    print(labels)
    timeseries_to_image(data, 'ECG-data of patient HR14099 in the PTBXL dataset', filename='/home/julian/Documents/projekt-master/bilder/ecg-example.png', show=True, save=True)
