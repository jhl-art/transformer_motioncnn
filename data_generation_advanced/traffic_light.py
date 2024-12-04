import numpy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class TrafficLight():
	def __init__(self):

		self.past_traffic_light_states = None
		self.past_traffic_light_valid = None 
		self.past_traffic_light_type = None
		self.current_traffic_light_state = None
		self.current_traffic_light_valid = None 
		self.current_traffic_light_type = None

		self.color_list = [
			np.array([255, 0, 0]),  #red
			np.array([255, 255, 0]), #green
			np.array([0, 255, 0]),  #yellow
			np.array([0, 0, 0]),   #off
			np.array([255, 125, 0]),  #flashing red
			np.array([125, 255, 0]),   # flashing yellow
			np.array([255, 255, 125]),  # flashing green
		]

	def parse_traffic_light(self, parsed_data):
		self.past_traffic_light_states, self.past_traffic_light_valid, self.past_traffic_light_type = self.get_history_traffic_light_state(parsed_data)
		self.current_traffic_light_state, self.current_traffic_light_valid, self.current_traffic_light_type = self.get_current_traffic_light_state(parsed_data)

	def get_history_traffic_light_state(self, parsed_data):
		self.past_traffic_light_states = tf.stack(
      			[parsed_data['traffic_light_state/past/x'], parsed_data['traffic_light_state/past/y']],-1).numpy()
		self.past_traffic_light_valid = parsed_data['traffic_light_state/past/valid'].numpy() > 0.0
		self.past_traffic_light_type =  parsed_data['traffic_light_state/past/state']
		return self.past_traffic_light_states, self.past_traffic_light_valid, self.past_traffic_light_type

	def get_current_traffic_light_state(self, parsed_data):
		self.current_traffic_light_state = tf.stack(
      			[parsed_data['traffic_light_state/current/x'], parsed_data['traffic_light_state/current/y']],-1).numpy()
		self.current_traffic_light_valid = parsed_data['traffic_light_state/current/valid'].numpy() > 0.0
		self.current_traffic_light_type = parsed_data['traffic_light_state/current/state']
		return self.current_traffic_light_state, self.current_traffic_light_valid, self.current_traffic_light_type

	def get_input_traffic_lights(self):
		input_traffic_light_state = np.concatenate((self.past_traffic_light_states, self.current_traffic_light_state), axis=0)
		input_traffic_light_valid = np.concatenate((self.past_traffic_light_valid, self.current_traffic_light_valid), axis=0)
		input_traffic_light_type = np.concatenate((self.past_traffic_light_type, self.current_traffic_light_type), axis=0)
		return input_traffic_light_state, input_traffic_light_valid, input_traffic_light_type

	def visualize_traffic_light(self, tl_state, tl_valid, tl_type, sdc):
		valid = (tl_valid > 0)
		valid_states = tl_state[valid]
		valid_types = tl_type[valid].reshape(-1,1)

		sdc_xy = sdc['xy']

		num_valid_tl,_ = valid_states.shape
		if num_valid_tl == 0:
			return

		for i in range(num_valid_tl):
			x,y = valid_states[i][0], valid_states[i][1]
			type = valid_types[i][0]

			index = (type % len(self.color_list))
			tl_color = self.color_list[index] / 255.0

			translated_x, translated_y = (x - sdc_xy[0]), (y - sdc_xy[1])
			rotated_x, rotated_y = np.array([translated_x, translated_x]) @ sdc['rotation_matrix'].T
			rotated_x += 250.0
			rotated_y += 250.0

			plt.scatter(rotated_x, rotated_y,  color=tl_color, zorder=10, s=4)
		return





