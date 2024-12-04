import os
import yaml
import tensorflow as tf
import argparse
from glob import glob
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import uuid
import torch

def get_config(path):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-path", type=str, required=True,
        help="Path to training data")
    parser.add_argument(
        "--val-data-path", type=str, required=True,
        help="Path to validation data")
    parser.add_argument(
        "--checkpoints-path", type=str, required=True,
        help="Path to checkpoints")
    parser.add_argument(
        "--config", type=str, required=True, help="Config file path")
    parser.add_argument("--multi-gpu", action='store_true')
    args = parser.parse_args()
    return args

def get_last_checkpoint_file(path):
    list_of_files = glob(f'{path}/*.pth')
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def create_figure_and_axes(size_pixels):
        """Initializes a unique figure and axes for plotting."""
        fig, ax = plt.subplots(1, 1, num=uuid.uuid4())
        # Sets output image to pixel resolution.
        dpi = 100
        size_inches = size_pixels / dpi
        fig.set_size_inches([size_inches, size_inches])
        fig.set_dpi(dpi)
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        ax.xaxis.label.set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='y', colors='black')
        fig.set_tight_layout(True)
        ax.grid(False)
        return fig, ax

def create_animation(images):
        plt.ioff()
        fig, ax = plt.subplots()
        dpi = 100
        size_inches = 1000 / dpi
        fig.set_size_inches([size_inches, size_inches])
        plt.ion()

        def animate_func(i):
            ax.set_axis_off()
            ax.imshow(images[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid('off')

        anim = animation.FuncAnimation(
              fig, animate_func, frames=len(images), interval=100)
        plt.close(fig)
        return anim

def dict_to_cuda(data):
    train_data = {}
    data_path = []
    for key, value in data.items():
        if isinstance(value, torch.Tensor) is False:
            data_path.append(value)
            continue
        train_data[key] = value.to('cuda')
    return train_data, data_path