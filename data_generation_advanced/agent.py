import numpy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Agent():
	def __init__(self):

		self.past_agent_states = None
		self.past_agent_states_valid = None
		self.current_agent_states = None 
		self.current_states_valid = None
		self.future_agent_states = None
		self.future_states_valid = None

		self.color_list = []
		for i in range(1, 44):
			self.color_list.append(np.array([i * 2, 0, 0]))
			self.color_list.append(np.array([0, i * 2, 0]))
			self.color_list.append(np.array([0, 0, i * 2]))

		self.agent_coorindates = {
			'min_xy': [],
			'center_xy': [],
			'max_xy': [],
			'width': 0.0
		}


	def parse_agent(self, parsed_data):
		self.past_agent_states, self.past_agent_states_valid = self.get_agent_history_state(parsed_data)
		self.current_agent_states, self.current_states_valid = self.get_agent_current_state(parsed_data)
		self.future_agent_states, self.future_states_valid = self.get_agent_future_state(parsed_data)
		self.agent_coorindates = self.get_agent_coorindates()

	def get_agent_history_state(self, parsed_data):
		# [num_agents, num_past_steps, 5] float32.
		self.past_agent_states = tf.stack(
		      [parsed_data['state/past/x'], parsed_data['state/past/y'], 
         	   parsed_data['state/past/bbox_yaw'], parsed_data['state/past/length'], 
               parsed_data['state/past/width']],-1).numpy()
		self.past_agent_states_valid = parsed_data['state/past/valid'].numpy() > 0.0
		return self.past_agent_states, self.past_agent_states_valid

	def get_agent_current_state(self, parsed_data):
		# [num_agents, 1, 5] float32.
		self.current_agent_states = tf.stack(
                [parsed_data['state/current/x'], parsed_data['state/current/y'], 
                 parsed_data['state/current/bbox_yaw'], parsed_data['state/current/length'],
		         parsed_data['state/current/width']],-1).numpy()
		self.current_states_valid = parsed_data['state/current/valid'].numpy() > 0.0
		return self.current_agent_states, self.current_states_valid

	def get_agent_future_state(self, parsed_data):
		# [num_agents, num_future_steps, 2] float32.
		self.future_agent_states = tf.stack(
      			[parsed_data['state/future/x'], parsed_data['state/future/y'], 
                 parsed_data['state/future/bbox_yaw'], parsed_data['state/future/length'],
                 parsed_data['state/future/width']],-1).numpy()
		self.future_states_valid = parsed_data['state/future/valid'].numpy() > 0.0
		return self.future_agent_states, self.future_states_valid

	def get_agent_coorindates(self):
     	# [num_agents, num_past_steps + 1 + num_future_steps] float32.
		return self.agent_coorindates

	def get_input_agent_state(self):
		input_agent_state = np.concatenate((self.past_agent_states, self.current_agent_states), axis=1)
		input_agent_valid = np.concatenate((self.past_agent_states_valid, self.current_states_valid), axis=1)
		return input_agent_state, input_agent_valid

	def get_future_agent_state(self):
		return self.future_agent_states, self.future_states_valid

	def visualize_agent_state(self, agent_state, agent_valid, sdc):
		valid = (agent_valid > 0.0)
		sdc_xy = sdc['xy']
		sdc_index = sdc['index']

		agent_x = (agent_state[:, 0] - sdc_xy[0]).reshape(-1,1)
		agent_y = (agent_state[:, 1] - sdc_xy[1]).reshape(-1,1)
		agent_xy = np.concatenate((agent_x, agent_y), axis=1)

		rotated_agent_xy = agent_xy @ sdc['rotation_matrix'].T
		rotated_agent_xy = rotated_agent_xy + np.array([250.0, 250.0])

		all_x = rotated_agent_xy[:, 0][valid]
		all_y = rotated_agent_xy[:, 1][valid]

		self.agent_coorindates['min_xy'] = [np.min(all_x), np.min(all_y)]
		self.agent_coorindates['max_xy'] = [np.max(all_x), np.max(all_y)]

		x_width = abs(250.0 - np.min(all_x)) + abs(250.0 - np.max(all_x))
		y_width = abs(250.0 - np.min(all_y)) + abs(250.0 - np.max(all_y))
		self.agent_coorindates['width'] = max(x_width, y_width)

		agent_yaw = agent_state[:, 2] - sdc['yaw']

		agent_length = agent_state[:, 3]
		agent_width = agent_state[:, 4]
		num_agent,_ = agent_state.shape

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
				plt.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'black', alpha=1, ms=2, linewidth=2, zorder=5)
			else:
				plt.fill(rotated_corners[:, 0], rotated_corners[:, 1], color, zorder=5)

		color_index = 0
		for i in range(num_agent):
			if valid[i] == False:
				continue
			color = self.color_list[color_index % len(self.color_list)]
			color_index+=1
			is_sdc = (sdc_index[0] == i)
			plot_bounding_box(rotated_agent_xy[i][0], rotated_agent_xy[i][1], agent_width[i], agent_length[i], agent_yaw[i], color, is_sdc)








