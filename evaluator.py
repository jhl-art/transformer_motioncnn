import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import pickle
import os
import matplotlib.animation as animation
from utils import create_figure_and_axes, create_animation
import matplotlib.pyplot as plt
import csv
import uuid

class Evaluator:
	def __init__(self, evaluator_result_path):
		self.evaluator_result_path = evaluator_result_path
		os.makedirs(evaluator_result_path, exist_ok=True)

		self.procecss_key_list = ['time_0', 'time_1', 'time_2', 'time_3', 
		                          'time_4', 'time_5', 
								  'time_6', 'time_7', 'time_8', 'time_9']

		self.color_list = ['red', 'orange', 'blue', 'yellow', 'pink', 'green']

	def calculate_top1_mse_and_fde(self, predicted_tensor, eval_data):

		trajectories = predicted_tensor['xy']  # Shape: [batch_size, n_modes, length, width]
		confidences = F.softmax(predicted_tensor['confidences'], dim=1)  # Normalize confidences

		# Step 1: Identify the index of the highest confidence mode for each sample
		max_indices = torch.argmax(confidences, dim=1)  # Shape: [batch_size]

    	# Step 2: Select the trajectories corresponding to the highest confidence
		batch_size, n_modes, length, width = trajectories.shape
		batch_indices = torch.arange(batch_size).unsqueeze(1).to(trajectories.device)  # Shape: [batch_size, 1]

   		# Expand max_indices to gather the highest-confidence trajectories
		max_indices_expanded = max_indices.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Shape: [batch_size, 1, 1, 1]
		max_indices_expanded = max_indices_expanded.expand(batch_size, 1, length, width)

    	# Gather the selected trajectories
		selected_trajectories = torch.gather(trajectories, dim=1, index=max_indices_expanded).squeeze(1)  # Shape: [batch_size, length, width]

    	# Ground truth trajectory
		ground_truth_trajectory = eval_data['gt_path']  # Shape: [batch_size, length, width]

    	# Step 3: Compute MSE (Mean Squared Error) over all time steps
		mse = torch.norm(selected_trajectories - ground_truth_trajectory, dim=(1, 2))  # Shape: [batch_size]
		mse = mse / (length * batch_size) # Average over the batch

		#print("mse: ", mse.item())

    	# Step 4: Compute FDE (Final Displacement Error) for the final time step
		final_ground_truth = ground_truth_trajectory[:, -1, :]  # Shape: [batch_size, width]
		final_selected_trajectory = selected_trajectories[:, -1, :]  # Shape: [batch_size, width]	
		fde = torch.norm(final_selected_trajectory - final_ground_truth, dim=1)  # Shape: [batch_size]
		fde = fde.mean()  # Average over the batch
	
		return mse, fde

	def calculate_multitrajectory_mse_and_fde(self, predicted_tensor, eval_data):
		
		trajectories = predicted_tensor['xy']  # (batch_size, 6, 80, 2)
		confidences = F.softmax(predicted_tensor['confidences'], dim=1) 

		batch_size, n_modes, length, width = trajectories.shape

		ground_truth_trajectory = eval_data['gt_path'] # (batch_size, 80, 2)
		ground_truth_trajectory = ground_truth_trajectory.unsqueeze(1)
		ground_truth_trajectory = ground_truth_trajectory.expand(batch_size,n_modes, length, width)

		difference = trajectories - ground_truth_trajectory
		
		# Step 3: Compute MSE (Mean Squared Error) over all time steps
		mse = torch.norm(trajectories - ground_truth_trajectory, dim=(2, 3))  # Shape: [batch_size, n_modes]
		weighted_mse = mse * confidences
		weighted_mse  = weighted_mse / (length * batch_size)

		# Step 4: Compute FDE (Final Displacement Error) for the final time step
		final_ground_truth = ground_truth_trajectory[:, :, -1, :]  # Shape: [batch_size, width]
		final_selected_trajectory = trajectories[:, :, -1, :]  # Shape: [batch_size, width]	
		fde = torch.norm(final_selected_trajectory - final_ground_truth, dim=2)  # Shape: [batch_size]
		weighted_fde = fde * confidences # Average over the batch
		return torch.min(weighted_mse), torch.min(weighted_fde)

	def dump_gif(self, image_sequence, output_dir):
		gif_name = "result.gif"
		anim = self.create_animation(image_sequence)
		writergif = animation.PillowWriter(fps=1)
		anim.save(os.path.join(output_dir, gif_name), writergif)
		print("save gif: ", os.path.join(output_dir, gif_name))
		return os.path.join(output_dir, gif_name)

	def create_animation(self, images):

		plt.ioff()
		fig, ax = plt.subplots(1, 1, num=uuid.uuid4())
		dpi = 100
		size_inches = 500 / dpi
		fig.set_size_inches([size_inches, size_inches])

		def animate_func(i):
			ax.set_axis_off()
			ax.imshow(images[i])
			ax.set_xticks([])
			ax.set_yticks([])
			ax.text(10, 20, f"Index: {i}", color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
			ax.grid('off')
			fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
			fig.tight_layout()

		anim = animation.FuncAnimation(fig, animate_func, frames=len(images), interval=100)
		plt.close(fig)
		return anim


	def create_image_sequence(self, data, top1_path):
		image_sequence = []

		for index in range(10):
			key = "time_" + str(index)
			image = data[key][0].cpu().numpy()
			image_sequence.append(image)

		last_image = data["time_10"][0].cpu().numpy()
		sdc_yaw = data['sdc_yaw'].cpu().numpy()
		gt_path = data['gt_path'][0].cpu().numpy()
		top1_path = top1_path.cpu().numpy()

		rotation_matrix = np.array(
				[[np.cos(-sdc_yaw), -np.sin(-sdc_yaw)],
				[np.sin(-sdc_yaw), np.cos(-sdc_yaw)]]
		)

		fig, ax = plt.subplots(figsize=(3, 3))
		plt.imshow(last_image)
		rotated_gt_path = gt_path @ rotation_matrix.T
		rotated_gt_path = np.reshape(rotated_gt_path, (80, 2))

		rotated_top1_path = top1_path @ rotation_matrix.T
		rotated_top1_path = np.reshape(rotated_top1_path, (80, 2))


		# plt.plot(gt_path[:, 0] + 250.0, gt_path[:, 1] + 250.0, color='black', linewidth=0.8, label='Ground Truth')
		# plt.plot(top1_path[:, 0] + 250.0, top1_path[:, 1] + 250.0, color='green', linewidth=0.8, label='Top 1')
		# plt.scatter(gt_path[:, 0] + 150.0, gt_path[:, 1] + 150.0, color='black', s=0.5, zorder=100)
		# plt.scatter(top1_path[:, 0] + 150.0, top1_path[:, 1] + 150.0, color='green', s=0.5, zorder=100)

		plt.plot(rotated_gt_path[:, 0] + 250.0, rotated_gt_path[:, 1] + 250.0, color='black', linewidth=0.8, zorder=9, label='Ground Truth')
		plt.plot(rotated_top1_path[:, 0] + 250.0, rotated_top1_path[:, 1] + 250.0, color='green', linewidth=0.5, zorder=10, label='Top 1')

		plt.axis('off')  # Optionally turn off axis if you want a clean image
		ax.grid('off')
		fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
		fig.tight_layout()
		plt.draw()

		# Step 2: Save the figure to a NumPy array
		canvas = plt.gca().figure.canvas
		canvas.draw()  # Update the canvas
		image_np = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
		image_np = image_np.reshape(300, 300, 3)

		plt.close()
		image_sequence.append(image_np)

		return image_sequence

	def visualize_result(self, predicted_tensor, eval_data):
		evaluator_visualization_path = self.evaluator_result_path 

		pickle_filename = eval_data['pickle_filename']
		last_two = pickle_filename[0].split("/")[-2:]

		folder1 = last_two[0]
		filename =  last_two[1]
		number = filename.split('.')[0]

		os.makedirs(evaluator_visualization_path + "/" + folder1, exist_ok=True)
		os.makedirs(evaluator_visualization_path + "/" + folder1 + "/" + str(number), exist_ok=True)

		dump_folder_path = evaluator_visualization_path + "/" + folder1 + "/" + str(number)


		trajectories = predicted_tensor['xy']  # Shape: [batch_size, n_modes, length, width]
		confidences = F.softmax(predicted_tensor['confidences'], dim=1)  # Normalize confidences

		# Step 1: Identify the index of the highest confidence mode for each sample
		max_indices = torch.argmax(confidences, dim=1)  # Shape: [batch_size]

    	# Step 2: Select the trajectories corresponding to the highest confidence
		batch_size, n_modes, length, width = trajectories.shape
		batch_indices = torch.arange(batch_size).unsqueeze(1).to(trajectories.device)  # Shape: [batch_size, 1]

   		# Expand max_indices to gather the highest-confidence trajectories
		max_indices_expanded = max_indices.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Shape: [batch_size, 1, 1, 1]
		max_indices_expanded = max_indices_expanded.expand(batch_size, 1, length, width)

    	# Gather the selected trajectories
		selected_trajectories = torch.gather(trajectories, dim=1, index=max_indices_expanded).squeeze(1)  # Shape: [batch_size, length, width]

		image_sequence = self.create_image_sequence(eval_data, selected_trajectories[0])
		print(len(image_sequence))
  
		gif_name = self.dump_gif(image_sequence, dump_folder_path)
  
		return gif_name

	def dump_top1_mse_and_losses(self, top1_mse, losses):
		
		filename = "top1mse_vs_loss.png"
		output_dir = self.evaluator_result_path + "/top1mse_vs_loss"
		os.makedirs(output_dir, exist_ok=True)
		plt.plot(top1_mse, losses, marker='o', color='b')
		plt.xlabel('top1 mse')
		plt.ylabel('loss')
		plt.title('XY Plot')
		plt.legend()
		plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

		min_value = min(top1_mse)
		max_value = max(top1_mse)
		mean_value = np.mean(np.array(top1_mse))

		with open(os.path.join(self.evaluator_result_path, "meta_top1mse.txt"), 'wb') as f:
			# Use an f-string to format the string and encode it to bytes
			f.write(f"Min/Mean/Max values: {min_value}, {mean_value}, {max_value}".encode('utf-8'))

	def dump_top1_fde_and_losses(self, top1_fde, losses):
		
		filename = "top1fde_vs_loss.png"
		output_dir = self.evaluator_result_path + "/top1fde_vs_loss"
		os.makedirs(output_dir, exist_ok=True)
		plt.plot(top1_fde, losses, marker='o', color='b')
		plt.xlabel('top1 fde')
		plt.ylabel('loss')
		plt.title('XY Plot')
		plt.legend()
		plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

		min_value = min(top1_fde)
		max_value = max(top1_fde)
		mean_value = np.mean(np.array(top1_fde))

		with open(os.path.join(self.evaluator_result_path, "meta_top1fde.txt"), 'wb') as f:
			# Use an f-string to format the string and encode it to bytes
			f.write(f"Min/Mean/Max values: {min_value}, {mean_value}, {max_value}".encode('utf-8'))

	def dump_multi_mse_and_losses(self, multi_mse, losses):
		
		filename = "top1fde_vs_loss.png"
		output_dir = self.evaluator_result_path + "/multi_mse_vs_loss"
		os.makedirs(output_dir, exist_ok=True)
		plt.plot(multi_mse, losses, marker='o', color='b')
		plt.xlabel('top1 fde')
		plt.ylabel('loss')
		plt.title('XY Plot')
		plt.legend()
		plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

		min_value = min(multi_mse)
		max_value = max(multi_mse)
		mean_value = np.mean(np.array(multi_mse))

		with open(os.path.join(self.evaluator_result_path, "meta_multimse.txt"), 'wb') as f:
			# Use an f-string to format the string and encode it to bytes
			f.write(f"Min/Mean/Max values: {min_value}, {mean_value}, {max_value}".encode('utf-8'))

	def dump_multi_fde_and_losses(self, multi_fde, losses):
		
		filename = "top1fde_vs_loss.png"
		output_dir = self.evaluator_result_path + "/multi_fde_vs_loss"
		os.makedirs(output_dir, exist_ok=True)
		plt.plot(multi_fde, losses, marker='o', color='b')
		plt.xlabel('top1 fde')
		plt.ylabel('loss')
		plt.title('XY Plot')
		plt.legend()
		plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

		min_value = min(multi_fde)
		max_value = max(multi_fde)
		mean_value = np.mean(np.array(multi_fde))

		with open(os.path.join(self.evaluator_result_path, "meta_multifde.txt"), 'wb') as f:
			# Use an f-string to format the string and encode it to bytes
			f.write(f"Min/Mean/Max values: {min_value}, {mean_value}, {max_value}".encode('utf-8'))



	def dump_csv_table(self, csv_table_2, csv_table_5, csv_table_8, csv_table_max, header_name):

		output_dir = self.evaluator_result_path + "/table/" + header_name
		os.makedirs(output_dir, exist_ok=True)

		def write_table(filename, table):
			with open(filename, "w", newline="") as f:
				writer = csv.writer(f)
				writer.writerow(["pickle_filename", "loss", header_name])  # Header
				writer.writerows(table)

		write_table(os.path.join(output_dir, "table_2.csv"), csv_table_2)
		write_table(os.path.join(output_dir, "table_5.csv"), csv_table_5)
		write_table(os.path.join(output_dir, "table_8.csv"), csv_table_8)
		write_table(os.path.join(output_dir, "table_max.csv"), csv_table_max)

