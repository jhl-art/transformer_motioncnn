import math
import os
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
os.environ['QT_SCALE_FACTOR'] = '1'
import uuid
import time

import matplotlib
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import HTML
import itertools
import tensorflow as tf
from glob import glob

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2


class DataGenerator():
	def __init__(self, data_folder) -> None:
		num_map_samples = 20000
		# Example field definition
		self.roadgraph_features = {
		    'roadgraph_samples/dir': tf.io.FixedLenFeature(
		        [num_map_samples, 3], tf.float32, default_value=None
		    ),
		    'roadgraph_samples/id': tf.io.FixedLenFeature(
		        [num_map_samples, 1], tf.int64, default_value=None
		    ),
		    'roadgraph_samples/type': tf.io.FixedLenFeature(
		        [num_map_samples, 1], tf.int64, default_value=None
		    ),
		    'roadgraph_samples/valid': tf.io.FixedLenFeature(
		        [num_map_samples, 1], tf.int64, default_value=None
		    ),
		    'roadgraph_samples/xyz': tf.io.FixedLenFeature(
		        [num_map_samples, 3], tf.float32, default_value=None
		    ),
		}
	    # Features of other agents.
		self.state_features = {
		    'state/id':
		        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
		    'state/type':
		        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
		    'state/is_sdc':
		        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
		    'state/tracks_to_predict':
		        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
		    'state/current/bbox_yaw':
		        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
		    'state/current/height':
		        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
		    'state/current/length':
		        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
		    'state/current/timestamp_micros':
		        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
		    'state/current/valid':
		        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
		    'state/current/vel_yaw':
		        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
		    'state/current/velocity_x':
		        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
		    'state/current/velocity_y':
		        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
		    'state/current/width':
		        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
		    'state/current/x':
		        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
		    'state/current/y':
		        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
		    'state/current/z':
		        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
		    'state/future/bbox_yaw':
		        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
		    'state/future/height':
		        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
		    'state/future/length':
		        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
		    'state/future/timestamp_micros':
		        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
		    'state/future/valid':
		        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
		    'state/future/vel_yaw':
		        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
		    'state/future/velocity_x':
		        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
		    'state/future/velocity_y':
		        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
		    'state/future/width':
		        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
		    'state/future/x':
		        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
		    'state/future/y':
		        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
		    'state/future/z':
		        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
		    'state/past/bbox_yaw':
		        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
		    'state/past/height':
		        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
		    'state/past/length':
		        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
		    'state/past/timestamp_micros':
		        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
		    'state/past/valid':
		        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
		    'state/past/vel_yaw':
		        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
		    'state/past/velocity_x':
		        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
		    'state/past/velocity_y':
		        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
		    'state/past/width':
		        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
		    'state/past/x':
		        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
		    'state/past/y':
		        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
		    'state/past/z':
		        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
		}

        # Features of traffic lights.
		self.traffic_light_features = {
		    'traffic_light_state/current/state':
		        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
		    'traffic_light_state/current/valid':
		        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
		    'traffic_light_state/current/x':
		        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
		    'traffic_light_state/current/y':
		        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
		    'traffic_light_state/current/z':
		        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
		    'traffic_light_state/past/state':
		        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
		    'traffic_light_state/past/valid':
		        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
		    'traffic_light_state/past/x':
		        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
		    'traffic_light_state/past/y':
		        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
		    'traffic_light_state/past/z':
		        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
		}

        # create feature description map
		self.features_description = {}
		self.features_description.update(self.roadgraph_features)
		self.features_description.update(self.state_features)
		self.features_description.update(self.traffic_light_features)

		self.input_files = [file for file in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, file))]
		self.output_files = []

		num_of_agent = 266
		num_of_roadgraph = 100
		num_of_traffic = 18

		self.agent_color_map = self.get_colormap(num_of_agent)
		self.roadgraph_color_map = self.get_colormap(num_of_roadgraph)
		self.traffic_color_map = self.get_colormap(num_of_traffic)
    
	def get_colormap(self, num_of_colors):
		"""Compute a color map array of shape [num_agents, 4]."""
		colors = cm.get_cmap('jet', num_of_colors)
		colors = colors(range(num_of_colors))
		np.random.shuffle(colors)
		return colors

	def get_input_files(self):
		return self.input_files

	def get_output_files(self):
		return self.output_files

	def get_feature_description(self):
		return self.features_description
