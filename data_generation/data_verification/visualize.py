import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
		ax.text(10, 20, f"Index: {i}", color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
		ax.grid('off')

	anim = animation.FuncAnimation(fig, animate_func, frames=len(images), interval=100)
	plt.close(fig)
	return anim

file_path = "raw/2.pkl"

# Open the file in 'rb' (read binary) mode
with open(file_path, 'rb') as file:
    data = pickle.load(file)

images = []
for i in range(10):
	key = "time_" + str(i)
	image = data[key]
	images.append(image)

print(data['gt_path'])

anim = create_animation(images)
writergif = animation.PillowWriter(fps=1)
gif_name = "test_raw2.gif"
anim.save(gif_name,writergif)
print("save gif: ", gif_name)
