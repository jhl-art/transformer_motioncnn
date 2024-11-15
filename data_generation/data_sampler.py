from waymo_open_dataset.protos import motion_metrics_pb2
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.metrics.ops import py_metrics_ops
from google.protobuf import text_format
from glob import glob
import tensorflow as tf
import itertools
from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import axis, cm
import matplotlib
import time
import uuid
import math
import os
import pickle 

os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
os.environ['QT_SCALE_FACTOR'] = '1'


def euclidean_distance(p, x, y):
	return math.sqrt((p[0] - x)**2 + (p[1] - y)**2)


class DataSampler():
	def __init__(self, data_path, index) -> None:
		self.data_path = data_path
		self.folder_index = index
		self.pickle_file_path = data_path + "/" + str(index) + ".pkl"
		self.parsed_data = {}
  
		self.past_agent_states = None
		self.past_agent_states_valid = None
		self.current_agent_states = None 
		self.current_states_valid = None
		self.future_agent_states = None
		self.future_states_valid = None
		self.road_graph_states = None
		self.road_graph_valid = None
		self.past_traffic_light_states = None
		self.past_traffic_light_valid = None
		self.current_traffic_light_state = None
		self.current_traffic_light_valid = None
		self.lane_id_type = dict()
		self.lane_id_to_xyz = dict()
  
		self.bottom_x, self.bottom_y = 0,0
		self.sdc_x, self.sdc_y = 0.0, 0.0
		self.sdc_index = None

	def get_record_data_path(self):
		return self.data_path
    
	def get_training_data_path(self):
		return self.pickle_file_path

	def parse_feature(self, data, features_description):
		self.parsed_data = tf.io.parse_single_example(data, features_description)
		self.get_agent_history_state()
		self.get_agent_current_state()
		self.get_agent_future_state()
		self.get_road_graph()
		self.get_history_traffic_light_state()
		self.get_current_traffic_light_state()
		self.get_sdc_position()
		return self.parsed_data

	def get_sdc_position(self):
		is_sdc = self.parsed_data['state/is_sdc'].numpy()
		sdc_index = np.squeeze(np.where(is_sdc == 1), axis=1)
		sdc_x = self.parsed_data['state/current/x'][sdc_index[0]][sdc_index[1]]
		sdc_y = self.parsed_data['state/current/y'][sdc_index[0]][sdc_index[1]]
		self.sdc_x = sdc_x.numpy()
		self.sdc_y = sdc_y.numpy()
		self.sdc_index = sdc_index
		return self.sdc_x, self.sdc_y, sdc_index
     
	def create_figure_and_axes(self, size_pixels):
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

	def fig_canvas_image(self,fig, ax):
		"""Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
		# Just enough margin in the figure to display xticks and yticks.
		fig.subplots_adjust(
		      left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
		ax.set_axis_off()
		fig.canvas.draw()
		data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
		return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

	def get_agent_history_state(self):
		# [num_agents, num_past_steps, 5] float32.
		self.past_agent_states = tf.stack(
		      [self.parsed_data['state/past/x'], self.parsed_data['state/past/y'], 
         	   self.parsed_data['state/past/bbox_yaw'],self.parsed_data['state/past/length'], 
               self.parsed_data['state/past/width']],-1).numpy()
		self.past_agent_states_valid = self.parsed_data['state/past/valid'].numpy() > 0.0
		return self.past_agent_states, self.past_agent_states_valid

	def get_agent_current_state(self):
		# [num_agents, 1, 5] float32.
		self.current_agent_states = tf.stack(
                [self.parsed_data['state/current/x'], self.parsed_data['state/current/y'], 
                 self.parsed_data['state/current/bbox_yaw'], self.parsed_data['state/current/length'],
		         self.parsed_data['state/current/width']],-1).numpy()
		self.current_states_valid = self.parsed_data['state/current/valid'].numpy() > 0.0
		return self.current_agent_states, self.current_states_valid

	def get_agent_future_state(self):
		# [num_agents, num_future_steps, 2] float32.
		self.future_agent_states = tf.stack(
      			[self.parsed_data['state/future/x'], self.parsed_data['state/future/y'], 
                 self.parsed_data['state/future/bbox_yaw'], self.parsed_data['state/future/length'],
                 self.parsed_data['state/future/width']],-1).numpy()
		self.future_states_valid = self.parsed_data['state/future/valid'].numpy() > 0.0
		return self.future_agent_states, self.future_states_valid

	def get_road_graph(self):
		# [num_points, 3] float32.
		valid = np.squeeze(self.parsed_data['roadgraph_samples/valid'].numpy() > 0.0)
		roadgraph_xyz = self.parsed_data['roadgraph_samples/xyz'].numpy()[valid]
		roadgraph_id = self.parsed_data['roadgraph_samples/id'].numpy()[valid]
		roadgraph_type = self.parsed_data['roadgraph_samples/type'].numpy()[valid]
  
		num_of_lane = roadgraph_id.shape[0]
		for id in roadgraph_id:
			self.lane_id_to_xyz[id[0]] = []
  
		for index in range(num_of_lane):
			id = roadgraph_id[index][0]
			lane_type = roadgraph_type[index]
			self.lane_id_type[id] = lane_type
			self.lane_id_to_xyz[id].append(roadgraph_xyz[index])
		return self.lane_id_type, self.lane_id_to_xyz

	def get_history_traffic_light_state(self):
		self.past_traffic_light_states = tf.stack(
      			[self.parsed_data['traffic_light_state/past/x'], self.parsed_data['traffic_light_state/past/y']],-1).numpy()
		self.past_traffic_light_valid = self.parsed_data['traffic_light_state/past/valid'].numpy() > 0.0
		self.past_traffic_light_type = self.parsed_data['traffic_light_state/past/state']
		return self.past_traffic_light_states, self.past_traffic_light_valid, self.past_traffic_light_type

	def get_current_traffic_light_state(self):
		self.current_traffic_light_state = tf.stack(
      			[self.parsed_data['traffic_light_state/current/x'], self.parsed_data['traffic_light_state/current/y']],-1).numpy()
		self.current_traffic_light_valid = self.parsed_data['traffic_light_state/current/valid'].numpy() > 0.0
		self.current_traffic_light_type = self.parsed_data['traffic_light_state/current/state']
		return self.current_traffic_light_state, self.current_traffic_light_valid, self.current_traffic_light_type

	def get_center_xy_and_width(self):
		# [num_agents, num_past_steps + 1 + num_future_steps] float32.
		all_states = np.concatenate([self.past_agent_states, self.current_agent_states, self.future_agent_states], 1)
		all_states_valid = np.concatenate([self.past_agent_states_valid, self.current_states_valid, self.future_states_valid], 1)

		valid_states = all_states[all_states_valid]
		all_y = valid_states[..., 1]
		all_x = valid_states[..., 0]

		center_y = (np.max(all_y) + np.min(all_y)) / 2
		center_x = (np.max(all_x) + np.min(all_x)) / 2

		range_y = np.ptp(all_y) # range of (x_min, x_max)
		range_x = np.ptp(all_x) # range of (y_min, y_max)

		width = max(range_y, range_x)
  
		self.bottom_x = np.min(all_x)
		self.bottom_y = np.min(all_y)
  
		return center_x, center_y, width

	def get_sdc_center_xy_and_width(self):
     	# [num_agents, num_past_steps + 1 + num_future_steps] float32.
		all_states = np.concatenate([self.past_agent_states, self.current_agent_states, self.future_agent_states], 1)
		all_states_valid = np.concatenate([self.past_agent_states_valid, self.current_states_valid, self.future_states_valid], 1)

		valid_states = all_states[all_states_valid]
		all_y = valid_states[..., 1]
		all_x = valid_states[..., 0]
  
		range_y = np.ptp(all_y) # range of (x_min, x_max)
		range_x = np.ptp(all_x) # range of (y_min, y_max)

		width = max(range_y, range_x)
		self.bottom_x = np.min(all_x)
		self.bottom_y = np.min(all_y)
  
		return self.sdc_x, self.sdc_y, width

	
	def get_bottom_xy(self):
		return self.bottom_x, self.bottom_y

	def plot_triangle(self, x, y, color, size=1):
		# Define the triangle's vertices relative to (x, y)
		triangle = [
			(x, y),  # First vertex at (x, y)
			(x + size, y),  # Second vertex
			(x + size/2, y + size * 0.866)  # Third vertex, height of equilateral triangle
		]
		# Unpack the triangle's x and y coordinates
		triangle_x, triangle_y = zip(*triangle)

		# Plot the triangle
		plt.fill(triangle_x, triangle_y, color)  # 'b' is the color blue

	def visualize_one_step_agent_states(self,agent_states, agent_valid, agent_color_map):
		agent_x = agent_states[:, 0]
		agent_y = agent_states[:, 1]
		agent_yaw = agent_states[:, 2]
		agent_length = agent_states[:, 3]
		agent_width = agent_states[:, 4]

		num_agents = len(agent_x)
		def plot_bounding_box(x_center, y_center, width, length, yaw, color, agent_is_sdc):
        	# Half width and height
			half_w, half_h = length / 2, width / 2
			corners = np.array([[-half_w, -half_h],[half_w, -half_h],[half_w, half_h],[-half_w, half_h]])
			rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],[np.sin(yaw), np.cos(yaw)]])
			# Rotate corners
			rotated_corners = np.dot(corners, rotation_matrix.T)
			# Shift to the actual center
			rotated_corners[:, 0] += x_center
			rotated_corners[:, 1] += y_center
			# Append the first corner to close the box (for plotting)
			rotated_corners = np.vstack([rotated_corners, rotated_corners[0]])
			# Plot the bounding box
			if agent_is_sdc == True:
				plt.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'black', alpha=1, ms=2, linewidth=2)
			else:
				plt.fill(rotated_corners[:, 0], rotated_corners[:, 1], color)

		for i in range(num_agents):
			if agent_valid[i] is False:
				continue
			color = agent_color_map[i]
			is_sdc = (self.sdc_index[0] == i)
			plot_bounding_box(agent_x[i], agent_y[i], agent_width[i], agent_length[i], agent_yaw[i], color, is_sdc)

	def visualize_one_step_traffic_light_state(self, traffic_light_state, valid, traffic_type, traffic_light_color_map):
		traffic_light_state = traffic_light_state[valid] 
		traffic_light_type = traffic_type[valid]

		num_of_traffic_light = len(traffic_light_state)
  
		def plot_triangle(x, y, color, size=1):
      		# Define the triangle's vertices relative to (x, y)
			triangle = [
				(x, y),  # First vertex at (x, y)
				(x + size, y),  # Second vertex
				(x + size/2, y + size * 0.866)  # Third vertex, height of equilateral triangle
			]
			# Unpack the triangle's x and y coordinates
			triangle_x, triangle_y = zip(*triangle)
			# Plot the triangle
			plt.fill(triangle_x, triangle_y, color)  # 'b' is the color blue
		for i in range(num_of_traffic_light):
			coord_x, coord_y, coord_type = traffic_light_state[i][0], traffic_light_state[i][1], traffic_light_type[i]
			plot_triangle(coord_x, coord_y, traffic_light_color_map[coord_type])

		return

	def create_animation(self, images):
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
		      fig, animate_func, frames=len(images) // 2, interval=100)
		plt.close(fig)
		return anim

	def increment_folder_index(self):
		self.folder_index += 1
		self.pickle_file_path = "processed_data/example_pkl/" + str(self.folder_index) + ".pkl"
		return

	def generate_training_data(self, roadgraph_color_map, agent_color_map, traffic_color_map):
		data = {}
		training_agent_state = np.concatenate((self.past_agent_states, self.current_agent_states), axis=1)
		training_agent_valid = np.concatenate((self.past_agent_states_valid, self.current_states_valid), axis=1)
    
		training_traffic_state = np.concatenate((self.past_traffic_light_states, self.current_traffic_light_state), axis=0)
		training_traffic_valid = np.concatenate((self.past_traffic_light_valid, self.current_traffic_light_valid), axis=0)
		training_traffic_type = np.concatenate((self.past_traffic_light_type, self.current_traffic_light_type), axis=0)
		
		center_x, center_y, width = self.get_sdc_center_xy_and_width()
		bottom_x, bottom_y = self.bottom_x, self.bottom_y

		num_agents, num_step, _ = training_agent_state.shape
		size = max(10, width * 1.0)
  
		#os.makedirs("processed_data/training_image/" + str(self.folder_index), exist_ok=True)
		images = []
		for time, (state, valid, traffic, traffic_valid, traffic_type) in enumerate(zip(np.split(training_agent_state, num_step, 1), 
                                                						  				np.split(training_agent_valid, num_step, 1), 
                                                        				  				np.split(training_traffic_state, num_step, 0), 
                                                              			 				np.split(training_traffic_valid, num_step, 0),
                                                                   		  				np.split(training_traffic_type, num_step, 0))):
			fig, ax = self.create_figure_and_axes(size_pixels=300)
            # visualize roadgraph 
			for lane_id, lane_type in self.lane_id_to_xyz.items():
				lane_type = self.lane_id_type[lane_id]
				lane_color = roadgraph_color_map[lane_type]
				lane_points = self.lane_id_to_xyz[lane_id]
				sorted_lane_points = np.array(sorted(lane_points, key=lambda x : euclidean_distance(x, bottom_x, bottom_y))).T
				ax.plot(sorted_lane_points[0, :], sorted_lane_points[1, :], lane_color, alpha=1, ms=1, linewidth=2)
			#visualize agent 
			self.visualize_one_step_agent_states(state[:,0], valid[:,0], agent_color_map)
            #visualize traffic state
			self.visualize_one_step_traffic_light_state(np.squeeze(traffic), np.squeeze(traffic_valid), np.squeeze(traffic_type),  traffic_color_map)
			ax.axis([-size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,size / 2 + center_y])
			ax.set_aspect('equal')
			ax.set_axis_off()
			image = self.fig_canvas_image(fig, ax)
			# images.append(image)
			# ax.imshow(image)
			# plt.savefig("processed_data/training_image/" + str(self.folder_index) + "/time_" + str(time) + ".png")
			data["time_" + str(time)] = image
			plt.close()

		# anim = self.create_animation(images)
		# writergif = animation.PillowWriter(fps=1)
		# gif_name = str(self.folder_index) + '_test_with_movie.gif'
		# anim.save(gif_name,writergif)
		# print("save gif: ", gif_name)
   
		future_relative_x = (self.future_agent_states[self.sdc_index[0], :, 0] - self.sdc_x).reshape(-1,1)
		future_relative_y = (self.future_agent_states[self.sdc_index[0], :, 1] - self.sdc_y).reshape(-1,1)
		future_position = np.concatenate((future_relative_x, future_relative_y), axis=1) #(80,1)
		data['gt_path'] = future_position
		data['data_path'] = self.data_path
		data['center_x'] = center_x
		data['center_y'] = center_y
		data['width'] = width
		print("finish dump pickle file: ", self.pickle_file_path)
		
		with open(self.pickle_file_path, 'wb') as file:
			pickle.dump(data, file)





























