import io
import os
from typing import Optional, Dict, Any

import numpy as np
import av
from PIL import Image
import torch
from matplotlib import gridspec
from torchvision.io import write_video
import matplotlib.pyplot as plt


class VideoWriter():
    def __init__(self, filename: str, fps: float, video_codec: str = "libx264", options: Optional[Dict[str, Any]] = None):
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

def timeseries_to_image(data: torch.Tensor, grad: torch.Tensor = None, pred_classes: list = None, downsample_factor=2, color=(0.1, 0.2, 0.5), convert_to_rgb=False, filename :str=None, verbose = True, show=False):
    batch, n, channels = data.shape
    rgba_colors = np.zeros((n, 4))
    for i in range(len(color)):
        rgba_colors[:, i] = color[i]
    rgba_colors[:, 3] = 1.0
    images = []
    for p, b in enumerate(data):
        fig, axs = plt.subplots(channels, 1, sharex='all', gridspec_kw={'hspace': 0}, figsize=(n/100/downsample_factor, channels*300/100/downsample_factor), dpi=100)
        fig.tight_layout(pad=0)
        x = np.arange(0, n)
        for i, ax in enumerate(axs):
            y = b[:, i]
            if not grad is None:
                alpha = grad[p, :, i]
                alpha = (alpha-grad[p, :, :].min())/(grad[p, :, :].max()-grad[p, :, :].min()) #TODO: consider normalizing this channelwise
                rgba_colors[:, 3] = alpha #alpha channel is gradient
            ax.scatter(x, y, label='channel_'+str(i), color=rgba_colors, s=1.0)
            #ax.label_outer()
        if not pred_classes is None:
            fig.suptitle(",".join([str(st) for st in pred_classes[p]]), fontsize=16)
        plt.legend()
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', dpi=100)
        if not filename is None:
            fig.savefig(filename + '_' + str(p) + '.png', dpi=100)
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        if convert_to_rgb:
            pil_img = Image.fromarray(img_arr, 'RGBA')
            temp = Image.new("RGB", pil_img.size, (255, 255, 255))
            temp.paste(pil_img, mask=pil_img.split()[3])
            img_arr = np.array(temp)  # 3 is the alpha channel

        images.append(img_arr)
        plt.close()
        if verbose:
            print("image {} of {} images complete".format(p+1, batch), flush=True)
        if show:
            plt.imshow(img_arr)
            plt.show()

    return np.stack(images)