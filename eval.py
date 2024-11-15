import numpy as np
import torch
torch.cuda.empty_cache()
import timm
import os
import argparse
from glob import glob
from tqdm import tqdm
from utils import get_config, get_last_checkpoint_file
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from image_cnn_net import SequentialMotionCNN
from loss import NLLGaussian2d
from postprocess import PostProcess
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
from train import MotionCNNDataset, parse_arguments
import matplotlib.pyplot as plt
import uuid
import torch.nn.functional as F
import matplotlib.animation as animation
np.set_printoptions(precision=5)

color_list = ['red', 'orange', 'blue', 'yellow', 'pink', 'green']

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

def dict_to_cuda(data):
    train_data = {}
    data_path = []
    for key, value in data.items():
        if isinstance(value, torch.Tensor) is False:
            print("not torch: ", key)
            data_path.append(value)
            continue
        train_data[key] = value.to('cuda')
    return train_data, data_path

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

def main():
	general_config = get_config("config/baseline.yaml")
	model_config = general_config['model']
	model = SequentialMotionCNN(model_config).to('cuda')
	loss_module = NLLGaussian2d().to('cuda')
	experiment_checkpoints_dir = "/mnt/data/nan_checkpoints/baseline/"
	file_name = "e61_b4700.pth"
	latest_checkpoint = os.path.join(experiment_checkpoints_dir, file_name)

	post_processer = PostProcess(model_config)
	procecss_key_list = ['time_0', 'time_1', 'time_2', 'time_3', 'time_4', 'time_5', 'time_6', 'time_7', 'time_8', 'time_9']

	gif_dir = '/mnt/data/image_cnn/generate_gif'
	os.makedirs(gif_dir, exist_ok=True)

	if latest_checkpoint is not None:
		print(f"Loading checkpoint from {latest_checkpoint}")
		checkpoint_data = torch.load(latest_checkpoint)
		model.load_state_dict(checkpoint_data['model_state_dict'])
		epochs_processed = checkpoint_data['epochs_processed']

	eval_dataset = MotionCNNDataset("/mnt/data/image_cnn/processed_data/example_pkl")
	eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

	for index, eval_data in enumerate(tqdm(eval_dataloader)):
		# center_x = eval_data['center_x']
		# center_y = eval_data['center_y']
		# width = eval_data['width']
		#size = max(10, width * 1.0)
		
		image_sequence = []
		for key in procecss_key_list:
			data_image = eval_data[key].detach().numpy().reshape(300,300,3)
			image_sequence.append(data_image)
		eval_data_, data_path = dict_to_cuda(eval_data)
		prediction_tensor = model(eval_data_)
		prediction_dict = post_processer.postprocess_predictions(
                prediction_tensor, model_config)

		fig, ax = create_figure_and_axes(300)
		last_image = eval_data_['time_10'].detach().cpu().numpy().reshape(300,300,3)
		plt.imshow(last_image)
		multipath = prediction_dict['xy'].detach().cpu().numpy().reshape(6,80,2)
		for i in range(6):
			path = multipath[i]
			plt.scatter(path[:, 0] + 150.0, path[:, 1] + 150.0, color=color_list[i], s=5)
		
		# Step 1: Render the figure
		plt.axis('off')  # Optionally turn off axis if you want a clean image
		plt.draw()

		# Step 2: Save the figure to a NumPy array
		canvas = plt.gca().figure.canvas
		canvas.draw()  # Update the canvas
		image_np = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
		image_np = image_np.reshape(300, 300, 3)

		# Optionally close the plot
		plt.close()
		image_sequence.append(image_np)


		anim =  create_animation(image_sequence)
		writergif = animation.PillowWriter(fps=1)
		gif_name = 'test_with_' + str(index) + '_movie.gif'
		anim.save(os.path.join(gif_dir, gif_name),writergif)
		print("save gif: ", gif_name)

		loss = loss_module(eval_data_, prediction_dict)
		print("loss: ", loss.item())




if __name__ == '__main__':
    main()

