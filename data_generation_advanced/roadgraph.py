import numpy
import numpy as np 
import tensorflow as tf
import math
from functools import cmp_to_key
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class Roadgraph():
	def __init__(self) -> None:
		# key: lane_id, value: Waypoint
		self.lane_id_to_waypoint = dict()
		self.lane_id_to_lane_type = dict()
		self.roadgraph_coordinates = dict()
		self.color_list = [
			np.array([255, 0, 0]),
			np.array([230, 0, 0]),
			np.array([205, 0, 0]),
			np.array([180, 0, 0]),
			np.array([155, 0, 0]),
			np.array([130, 0, 0]),
			np.array([105, 0, 0]),
			np.array([0, 255, 0]),
			np.array([0, 230, 0]),
			np.array([0, 205, 0]),
			np.array([0, 180, 0]),
			np.array([0, 155, 0]),
			np.array([0, 130, 0]),
			np.array([0, 0, 255]),
			np.array([0, 0, 230]),
			np.array([0, 0, 205]),
			np.array([0, 0, 180]),
			np.array([0, 0, 155]),
			np.array([0, 0, 130]),
			np.array([0, 0, 105]),
		]

	class Waypoint:
		def __init__(self, rotated_xy, heading_value):
			self.xy_value = rotated_xy
			self.heading_value = heading_value

		def xy(self):
			return self.xy_value

		def heading(self):
			return self.heading_value

		def debug_string(self):
			return str(self.xy_value) + " " + str(self.heading_value)

	def parse_roadgraph(self, parsed_data, sdc):

		sdc_xy = sdc['xy']
		sdc_yaw = sdc['yaw']

		# [num_points, 3] float32.
		valid = np.squeeze(parsed_data['roadgraph_samples/valid'].numpy() > 0.0)
		# [valid, 3]
		roadgraph_sample_xyz = parsed_data['roadgraph_samples/xyz'].numpy()[valid]
		roadgraph_sample_id = parsed_data['roadgraph_samples/id'].numpy()[valid]
		roadgraph_sample_type = parsed_data['roadgraph_samples/type'].numpy()[valid]
		roadgraph_sample_dir = parsed_data['roadgraph_samples/dir'].numpy()[valid]

		def get_theta(heading, sdc_heading):
			if np.linalg.norm(heading) == 0 or np.linalg.norm(sdc_heading) == 0:
				return 0.0
			dot_product = np.dot(heading, sdc_heading)
			cos_theta = dot_product / (np.linalg.norm(heading) * np.linalg.norm(sdc_heading))
			return cos_theta

		sdc_heading = [np.cos(sdc_yaw), np.sin(sdc_yaw), 0]

		self.roadgraph_xyz = roadgraph_sample_xyz

		num_lanes, _ = roadgraph_sample_id.shape

		for i in range(num_lanes):
			lane_id = roadgraph_sample_id[i][0]
			self.lane_id_to_waypoint[lane_id] = []
			lane_type = roadgraph_sample_type[i]
			self.lane_id_to_lane_type[lane_id] = lane_type

		min_x, min_y = 1000000, 1000000
		max_x, max_y = -1000000,-1000000

		for i in range(num_lanes):
			lane_id = roadgraph_sample_id[i][0]
			xy = np.array([roadgraph_sample_xyz[i][0], roadgraph_sample_xyz[i][1]]) - np.array([sdc_xy[0], sdc_xy[1]])
			rotated_xy = xy @ sdc['rotation_matrix'].T
			rotated_xy = rotated_xy + np.array([250.0, 250.0])

			min_x = min(rotated_xy[0], min_x)
			min_y = min(rotated_xy[1], min_y)
			max_x = max(rotated_xy[0], max_x)
			max_y = max(rotated_xy[1], max_y)

			heading = roadgraph_sample_dir[i]
			heading[2] = 0
			relative_angles = (get_theta(heading, sdc_heading) + np.pi) % (2 * np.pi) - np.pi
			waypoint = self.Waypoint(rotated_xy, relative_angles)
			self.lane_id_to_waypoint[lane_id].append(waypoint)

		x_width = abs(250.0 - min_x) + abs(250.0 - max_x)
		y_width = abs(250.0 - min_y) + abs(250.0 - max_y)

		self.roadgraph_coordinates['min_xy'] = np.array([min_x, min_y])
		self.roadgraph_coordinates['max_xy'] = np.array([max_x, max_y])
		self.roadgraph_coordinates['width'] = max(x_width, y_width)
		return

	def get_roadgraph_coordinates(self):
		return self.roadgraph_coordinates


	def sort_lane_waypoint(self, bottom_x=0.0, bottom_y=0.0, bottom_z=0.0):
		# bottom_waypoint = self.Waypoint([bottom_x, bottom_y, bottom_z], [0,0,0])
		
		# def euclidean_distance(bottom_waypoint, waypoint):
		# 	bottom_xyz = bottom_waypoint.xyz()
		# 	waypoint_xyz = waypoint.xyz()
		# 	return math.sqrt((bottom_xyz[0] - waypoint_xyz[0])**2 + (bottom_xyz[1] - waypoint_xyz[1])**2 + (bottom_xyz[2] - waypoint_xyz[2])**2)

		def x_and_y_comparator(waypoint1, waypoint2):
			xyz1 = waypoint1.xy()
			xyz2 = waypoint2.xy()
			if xyz1[0] != xyz2[0]:
				return -1 if xyz1[0] < xyz2[0] else 1
			if xyz1[1] != xyz2[1]:
				return -1 if xyz1[1] < xyz2[1] else 1
			return 0

		for key in self.lane_id_to_waypoint.keys():
			lane_id_to_waypoint = self.lane_id_to_waypoint[key]
			#lane_id_to_waypoint = np.array(sorted(lane_id_to_waypoint, key=lambda x: euclidean_distance(bottom_waypoint, x))).T
			lane_id_to_waypoint = np.array(sorted(lane_id_to_waypoint, key=cmp_to_key(x_and_y_comparator))).T
			self.lane_id_to_waypoint[key] = lane_id_to_waypoint
		return

	def visualize_roadgraph(self):
		segments = []
		segment_colors = []

		for key,waypoints in self.lane_id_to_waypoint.items():
			lane_id = key
			lane_type = self.lane_id_to_lane_type[key]
			lane_color = self.color_list[(lane_type.item() % 20)] / 255.0

			xy = np.array([wp.xy() for wp in waypoints])
			relative_angles = np.array([wp.heading() for wp in waypoints])
			# Map angles to lightness (0 to 1)
			lightness = (np.abs(relative_angles) / np.pi)  # Normalize absolute angles to [0, 1]
			# Adjust color lightness by mixing with white
			colors = lane_color * (1 - lightness[:, np.newaxis]) + np.ones(3) * lightness[:, np.newaxis]
			for i in range(len(xy) - 1):
				segment = xy[i:i + 2, :2]  # Extract x, y for the line segment
				segments.append(segment)
				segment_colors.append(colors[i])  # Assign color to the segment
				#plt.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=0.2, zorder=0)

		line_collection = LineCollection(segments, colors=segment_colors, linewidths=0.5, zorder=0)
		plt.gca().add_collection(line_collection)
		return












