from waymo_open_dataset.protos import motion_metrics_pb2
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.metrics.ops import py_metrics_ops
from google.protobuf import text_format
from roadgraph import Roadgraph
from agent import Agent
from traffic_light import TrafficLight
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
import datetime

os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
os.environ['QT_SCALE_FACTOR'] = '1'

class DataSampler():
	def __init__(self, output_folder, pickle_index) -> None:
		pkl_name = str(pickle_index) + ".pkl"
		pkl_file = os.path.join(output_folder, pkl_name)

		image_folder_name = "image_" + str(pickle_index)
		self.image_folder = os.path.join(output_folder, image_folder_name)
		#os.makedirs(self.image_folder, exist_ok=True)

		# The place to dump generated image data for training, validation and testing.
		self.dump_pkl_file = pkl_file
		# data parsed from tfrecord
		self.parsed_data = {}
		# dictionary contains sdc['xy'], sdc['yaw'], sdc['index'] = index agent is sdc
		self.sdc = {}
		# class to parse and visualize roadgraph, agents information 
		self.roadgraph_raster = Roadgraph()
		self.agent_raster = Agent()
		self.traffic_light_raster = TrafficLight()

		# record min,max,center values in roadgraph, agents
		self.road_graph = dict()
		self.agent_graph = dict()

	def parse_feature(self, tfrecord, features_description):
		if os.path.exists(self.dump_pkl_file):
			print(self.dump_pkl_file, " already exists")
			return
		self.parsed_data = tf.io.parse_single_example(tfrecord, features_description)
		self.sdc = self.extract_sdc_information(self.parsed_data)
		self.roadgraph_raster.parse_roadgraph(self.parsed_data, self.sdc)
		self.agent_raster.parse_agent(self.parsed_data)
		self.traffic_light_raster.parse_traffic_light(self.parsed_data)

		self.road_graph = self.roadgraph_raster.get_roadgraph_coordinates()

		bottom_x, bottom_y, bottom_z = self.road_graph['min_xy'][0], self.road_graph['min_xy'][1], 0
		self.roadgraph_raster.sort_lane_waypoint(bottom_x, bottom_y, bottom_z)

	def extract_sdc_information(self, parsed_data):
		'''
		Return informations related to sdc postions, yaw and index.
		'''
		is_sdc = self.parsed_data['state/is_sdc'].numpy()
		sdc_index = np.squeeze(np.where(is_sdc == 1), axis=1)

		sdc_x = self.parsed_data['state/current/x'][sdc_index[0]][sdc_index[1]]
		sdc_y = self.parsed_data['state/current/y'][sdc_index[0]][sdc_index[1]]
		sdc_yaw = self.parsed_data['state/current/bbox_yaw'][sdc_index[0]][sdc_index[1]]

		sdc = {
			'xy': [sdc_x.numpy().item(), sdc_y.numpy().item()],
			'yaw': sdc_yaw.numpy().item(),
			'index': sdc_index
		}


		rotation_matrix = np.array(
			[[np.cos(-sdc['yaw']), -np.sin(-sdc['yaw'])],
			[np.sin(-sdc['yaw']), np.cos(-sdc['yaw'])]]
		)

		sdc['rotation_matrix'] = rotation_matrix

		return sdc

	def center_axis_on_image(self, fig, ax, center_xy, width, image_name=None):
		center_x, center_y = center_xy[0], center_xy[1]
		ax.axis([ -width / 2 + center_x, width / 2 + center_x, -width / 2 + center_y, width / 2 + center_y])
		ax.set_aspect('equal')
		ax.set_axis_off()
		image = self.fig_canvas_image(fig, ax)
		# if image_name:
		# 	ax.imshow(image)
		# 	plt.savefig(os.path.join(self.image_folder, image_name))
		# 	print("successfully dump image: ", os.path.join(self.image_folder, image_name))
		# 	plt.close()
		return image

	def create_figure_and_axes(self, size_pixels):
		"""Initializes a unique figure and axes for plotting."""
		fig, ax = plt.subplots(1, 1, num=uuid.uuid4())
		# Sets output image to pixel resolution.
		dpi = 100
		size_inches = size_pixels / dpi
		fig.set_size_inches([size_inches, size_inches])
		fig.set_dpi(dpi)  #how many pixel patches to this size inches
		fig.set_facecolor('white')
		ax.set_facecolor('white')
		ax.xaxis.label.set_color('black')
		ax.tick_params(axis='x', colors='black')
		ax.yaxis.label.set_color('black')
		ax.tick_params(axis='y', colors='black')
		fig.set_tight_layout(True)
		ax.grid(False)
		return fig, ax

	def fig_canvas_image(self, fig, ax):
		"""Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
		# Just enough margin in the figure to display xticks and yticks.
		fig.subplots_adjust(
		      left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
		ax.set_axis_off()
		fig.canvas.draw()
		data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
		return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


	def generate_input_data(self):

		if os.path.exists(self.dump_pkl_file):
			return
		
		data = {}
		images = []
		# (128,11,5), (128,11)
		input_agent_states, input_agent_valid = self.agent_raster.get_input_agent_state()
		input_traffic_light_states, input_traffic_light_valid, input_traffic_light_type = self.traffic_light_raster.get_input_traffic_lights()
		num_agents, num_step, _ = input_agent_states.shape

		for time, (agent_state, agent_valid, tl_state, tl_valid, tl_type) in enumerate(zip(np.split(input_agent_states, num_step, 1), 
															  							   np.split(input_agent_valid,  num_step, 1),
															  							   np.split(input_traffic_light_states, num_step, 0),
															  							   np.split(input_traffic_light_valid, num_step, 0),
															  							   np.split(input_traffic_light_type, num_step, 0))):
			fig, ax = self.create_figure_and_axes(size_pixels=500)
			self.roadgraph_raster.visualize_roadgraph()
			self.agent_raster.visualize_agent_state(agent_state[:,0], agent_valid[:,0], self.sdc)
			agent_graph = self.agent_raster.get_agent_coorindates()
			self.traffic_light_raster.visualize_traffic_light(tl_state, tl_valid, tl_type, self.sdc)
			image_name = "time_" + str(time) + ".png"
			image = self.center_axis_on_image(fig, ax, [250.0,  250.0], agent_graph['width'] * 2, None)
			#images.append(image)
			data["time_" + str(time)] = image

			plt.close(fig)

		sdc_index = self.sdc['index']
		sdc_x, sdc_y = self.sdc['xy'][0], self.sdc['xy'][1]

		future_agent_states, future_agent_valid = self.agent_raster.get_future_agent_state()
		future_relative_x = (future_agent_states[sdc_index[0], :, 0] - sdc_x).reshape(-1,1)
		future_relative_y = (future_agent_states[sdc_index[0], :, 1] - sdc_y).reshape(-1,1)
		future_position = np.concatenate((future_relative_x, future_relative_y), axis=1) #(80,1)

		data['gt_path'] = future_position
		data['sdc_yaw'] = self.sdc['yaw']

		with open(self.dump_pkl_file, 'wb') as file:
			pickle.dump(data, file)

		print("successfully dumped pickle file: ", self.dump_pkl_file)

		
			












