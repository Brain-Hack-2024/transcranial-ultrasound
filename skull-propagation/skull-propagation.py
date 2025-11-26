# %%
import numpy as np
import jax
from functools import partial
from jax import jit
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt

from jwave import FiniteDifferences, FourierSeries
from jwave.geometry import Domain, Medium, circ_mask
from jwave.utils import display_complex_field, show_positive_field

import sys
sys.path.append('../') # add parent directory to sys path

# key = random.PRNGKey(42)  # Random seed


# %%

# define linear ultrasound transducer (P4-1)
nelements = 256
element_pitch = 2.95e-4 # distance between transducer elements
transducer_extent = (nelements - 1) * element_pitch # length of the transducer [m]
transducer_frequency = 1e6 # frequency of the transducer [Hz]
omega = 2 * np.pi * transducer_frequency # angular
transducer_magnitude = 1e6 # magnitude of the transducer [Pa]
print(f"Transducer extent: {transducer_extent:.3f} m")

# %%
import pydicom
import matplotlib.pyplot as plt

# Read the DICOM file
dcm_file = pydicom.dcmread("Skull Z_1201.dcm")

# Extract the pixel data
pixel_data = dcm_file.pixel_array

# %%
skull_piece = pixel_data[:250,600:1200]>20000
# Flip the skull piece on the y-axis
skull_piece = np.flipud(skull_piece)

# Add more rows of zeros at the top of the skull_piece
num_rows_to_add = 100  # You can adjust this number as needed
additional_rows = np.zeros((num_rows_to_add, skull_piece.shape[1]), dtype=skull_piece.dtype)
skull_piece = np.vstack((additional_rows, skull_piece))


plt.imshow(skull_piece, cmap="gray")
plt.gca().invert_yaxis()

# %%

from utils.jwave_utils import get_domain

# define spatial parameters
N = np.array(skull_piece.shape).astype(int) # grid size [grid points]
dx = np.array([9.07935931401377e-5, 9.07935931401377e-5]) # grid spacing [m]
pml = np.array([20, 20]) # size of the perfectly matched layer [grid points]

domain = get_domain(N, dx)

# %%

# define real transducer positions
transducer_width = pml[0] # width of the transducer [grid points]
transducer_y_start = N[1]//2 - nelements//2 # start index of the transducer in the y-direction [grid points]
element_positions = np.array([
    N[0] - 1 - transducer_width * np.ones(nelements),
    np.linspace(transducer_y_start, transducer_y_start + nelements - 1, nelements)
], dtype=int)

# %%

# get medium
speed_background = 1550
density_background = 1000

speed_skull = 2700
density_skull = 1800

speed_map = np.where(skull_piece == 1, speed_skull, speed_background)
density_map = np.where(skull_piece == 1, density_skull, density_background)

medium = Medium(
    domain,
    speed_map,
    density_map,
    pml_size=pml[0],
)

# %%
ext = [0, N[1]*dx[1], N[0]*dx[0], 0]
plt.scatter(element_positions[1]*dx[1], element_positions[0]*dx[0],
            c='r', marker='o', s=5, label='transducer element')
plt.imshow(medium.sound_speed, cmap='gray', extent=ext)
plt.colorbar(label='Speed of sound [m/s]')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.legend(prop={'size': 7})
plt.gca().invert_yaxis()
plt.show()

# %%

from utils.jwave_utils import get_plane_wave_excitation
from jwave.geometry import TimeAxis

time_axis = TimeAxis.from_medium(medium, cfl=0.3)
angle = 0 * np.pi / 180
sources, signal, carrier_signal = get_plane_wave_excitation(domain, time_axis, transducer_magnitude, transducer_frequency, element_pitch, element_positions, angle=angle)

# %%
plt.plot(sources.signals[10])
plt.xlabel('Time point')
plt.ylabel('Amplitude [Pa]')
plt.show()
# %%
from utils.jwave_utils import get_data

# simulate data using jwave
pressure, _ = get_data(medium.sound_speed, medium.density, domain, time_axis, sources, element_positions)
#%%
data = pressure.params
skull_mask = medium.sound_speed != speed_background

# Create a mask that zeros out the skull and everything below it
# Find the topmost skull pixel in each column
mask = np.ones_like(skull_mask, dtype=float)
for col in range(skull_mask.shape[1]):
    skull_indices = np.where(skull_mask[:, col])[0]
    if len(skull_indices) > 0:
        # Zero out from the first skull pixel to the bottom
        first_skull_row = skull_indices[0]
        mask[first_skull_row:, col] = 0

# Apply the mask to the data (broadcast mask to match data shape)
# data has shape (time, height, width, channels), mask has shape (height, width)
masked_data = data * mask[np.newaxis, :, :, np.newaxis]

fig = plt.figure(frameon=False)
ax = fig.add_axes([0, 0, 1, 1])  # [left, bottom, width, height] - fills entire figure
ax.set_axis_off()

# Create inverted berlin colormap (RGB complement - color wheel opposite)
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# Get the berlin colormap and invert RGB values (1 - RGB)
berlin_cmap = cm.get_cmap('managua')
berlin_colors = berlin_cmap(np.linspace(0, 1, 256))
inverted_colors = berlin_colors.copy()
inverted_colors[:, :3] = 1 - berlin_colors[:, :3]  # Invert RGB, keep alpha
inverted_berlin = LinearSegmentedColormap.from_list('inverted_berlin', inverted_colors)

# Plot only the wave field (no background) with RGB-inverted colormap
im = ax.imshow(masked_data[1000], cmap='berlin', vmin=data.min()/2, vmax=data.max()/2)
plt.colorbar(im)

# %%

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def create_pressure_animation(data, skull_mask, output_filename='pressure_animation.mp4', duration=10, fps=24):
    # Create a mask that zeros out the skull and everything below it
    mask = np.ones_like(skull_mask, dtype=float)
    for col in range(skull_mask.shape[1]):
        skull_indices = np.where(skull_mask[:, col])[0]
        if len(skull_indices) > 0:
            # Zero out from the first skull pixel to the bottom
            first_skull_row = skull_indices[0]
            mask[first_skull_row:, col] = 0

    # Apply the mask to the data (broadcast mask to match data shape)
    masked_data = data * mask[np.newaxis, :, :, np.newaxis]

    # Get the data dimensions to set figure size appropriately
    height, width = masked_data.shape[1], masked_data.shape[2]
    dpi = 300  # Increased DPI for higher resolution (was 100)
    # Scale up the figure size for better quality
    scale_factor = 3  # Makes the output 3x larger
    figsize = (width * scale_factor / dpi, height * scale_factor / dpi)

    # Set up the figure to be full-screen and eliminate white margins
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])  # [left, bottom, width, height] - fills entire figure
    ax.set_axis_off()

    # Use managua colormap (no inversion needed based on static figure)
    # Find global min and max for consistent colorscale (using original data for colorscale)
    vmin, vmax = data.min() / 2, data.max() / 2

    # Create the initial pressure plot with flipped y-axis
    im = ax.imshow(masked_data[0][::-1], cmap='berlin', vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    # Calculate the number of frames for the movie
    total_frames = data.shape[0]
    frames_to_show = duration * fps
    frame_skip = max(1, total_frames // frames_to_show)

    # Set up the animation
    def animate(i):
        frame = i * frame_skip
        im.set_array(masked_data[frame][::-1])  # Flip the y-axis for each frame
        return [im]

    # Create the animation
    anim = animation.FuncAnimation(fig, animate, frames=frames_to_show, interval=1000/fps, blit=True)

    # Save the animation without showing the colorbar, title, or axes
    anim.save(output_filename, fps=fps, writer='ffmpeg')

    plt.close(fig)  # Close the figure to free up memory

    print(f"Animation saved as '{output_filename}' ({duration} seconds duration, {fps} fps, showing every {frame_skip} frame(s))")
# %%
p = pressure.params[:pressure.params.shape[0]//2]
create_pressure_animation(p, skull_mask, output_filename=f'pressure_animation_{transducer_frequency/1e6:.1f}MHz_masked.mp4', duration=5)
# %%
