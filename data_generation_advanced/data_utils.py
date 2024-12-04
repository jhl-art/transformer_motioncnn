import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import axis, cm
import matplotlib
import uuid
import numpy as np

def create_figure_and_axes(size_pixels):
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

def fig_canvas_image(fig, ax):
		"""Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
		# Just enough margin in the figure to display xticks and yticks.
		fig.subplots_adjust(
		      left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
		ax.set_axis_off()
		fig.canvas.draw()
		data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
		return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

