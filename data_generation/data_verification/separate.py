from PIL import Image
import os

# Load the GIF
gif_path = 'test8.gif'
output_folder = 'generated_image_test8'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the GIF file
gif = Image.open(gif_path)

# Extract frames
frame_index = 0
while True:
    # Save current frame as PNG
    frame_path = os.path.join(output_folder, f"frame_{frame_index:03d}.png")
    gif.save(frame_path)

    frame_index += 1
    try:
        # Move to the next frame
        gif.seek(gif.tell() + 1)
    except EOFError:
        # End of frames
        break

print(f"Extracted {frame_index} frames to '{output_folder}'")
