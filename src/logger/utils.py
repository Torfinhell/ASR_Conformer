import io

import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import ToTensor

import torch
import numpy as np
plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop


def plot_images(imgs, config):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W).
        config (DictConfig): hydra experiment config.
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    # name of each img in the array
    names = config.writer.names
    # figure size
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        # channels must be in the last dim
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")  # we do not need axis
    # To create a tensor from matplotlib,
    # we need a buffer to save the figure
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    # convert buffer to Tensor
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image

def melForward(f):
    return 2595*np.log10(1 + f/700) #wikipedia suggestion
def melInverse(m):
    return (10**(m/2595)-1)*700
def plot_spectrogram(spectrogram, config, name=None):
    """
    Plot spectrogram

    Args:
        spectrogram (Tensor): spectrogram tensor.
        name (None | str): optional name.
    Returns:
        image (Image): image of the spectrogram
    """
    f,ax = plt.subplots(figsize=(26, 7))
    mel_spec_config=config.transforms.instance_transforms.train.get_spectrogram
    hop_length=mel_spec_config["hop_length"]
    n_mels=mel_spec_config["n_mels"]
    sample_rate=mel_spec_config["sample_rate"]
    tGrid=np.arange(0,spectrogram.shape[1])*hop_length/sample_rate
    fGrid=np.arange(n_mels)
    tt, ff=np.meshgrid(tGrid, fGrid)
    im = ax.pcolormesh(tt, ff, 20* np.log10(spectrogram+1e-8), cmap="gist_heat")
    ax.set_xlabel('Time, sec', size=20)
    ax.set_ylabel('Frequency, MelID', size=20)
    f.colorbar(im)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    # convert buffer to Tensor
    image = ToTensor()(PIL.Image.open(buf))
    plt.close()
    return image
