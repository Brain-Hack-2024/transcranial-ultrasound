#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[140]:


import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import jit
from jax.lib import xla_bridge
print(f"Jax is using: {xla_bridge.get_backend().platform}")


# In[141]:


import torch


# # Setup

# ## Transducer

# In[142]:


# define linear ultrasound transducer (P4-1)
nelements = 64
element_pitch = 2.95e-4 # distance between transducer elements
transducer_extent = (nelements - 1) * element_pitch # length of the transducer [m]
transducer_frequency = 1e6 # frequency of the transducer [Hz]
transducer_magnitude = 1e6 # magnitude of the transducer [Pa]
print(f"Transducer extent: {transducer_extent:.3f} m")


# ## Domain

# In[143]:


# define spatial domain
N = np.array([128, 128]).astype(int) # grid size [grid points]
dx = np.array([element_pitch, element_pitch]) # grid spacing [m]
extent = N * dx # grid size [m]
pml = np.array([20, 20]) # size of the perfectly matched layer [grid points]
print(f"Number of grid points: {N}\nGrid size: {extent} m\nGrid spacing: {dx} m")
assert transducer_extent < extent[0] - 2*pml[0]*dx[0], "Transducer extent is larger than grid size"

from jwave.geometry import Domain
domain = Domain((N[0],N[1]), (dx[0], dx[1]))


# In[144]:


# define transducer positions in spatial domain
transducer_depth = pml[1] # depth of the transducer [grid points]
transducer_x_start = N[0]//2 - nelements//2 # start index of the transducer in the x-direction [grid points]
element_positions = np.array([
    np.linspace(transducer_x_start, transducer_x_start + nelements - 1, nelements),
    (N[1] - transducer_depth) * np.ones(nelements),
], dtype=int)


# ## Acoustic medium

# In[382]:


np.random.seed(28)

# define reference properties
c0 = 1500 # reference speed of sound [m/s]
rho0 = 1000 # reference density [kg/m^3]

# define a random distribution of scatterers for the medium
background_map_mean = 1
background_map_std = 0.004#8
background_map = background_map_mean + background_map_std * np.random.randn(N[0], N[1])
sound_speed = c0 * np.ones(N) * background_map
density = rho0 * np.ones(N) * background_map

# define highly scattering region
scatterer_radius = 2 # radius of scatterers [grid points]
scatterer_contrast = 1.1 # contrast of scatterers
scatterer_positions = np.array([[N[0]//2, N[1]//2]], dtype=int)
scatterer_map = np.zeros(N)
x, y = np.ogrid[:N[0], :N[1]]
for scatterer_position in scatterer_positions:
    scatterer_map[(x - scatterer_position[0])**2 + (y - scatterer_position[1])**2 <= (scatterer_radius)**2] = 1
sound_speed[scatterer_map == 1] = c0*scatterer_contrast
density[scatterer_map == 1] = rho0*scatterer_contrast

# define medium
from jwave import FourierSeries
from jwave.geometry import Medium
sound_speed = FourierSeries(np.expand_dims(sound_speed, -1), domain)
density = FourierSeries(np.expand_dims(density, -1), domain)
medium = Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=pml[0])

homogeneous_background_map = np.ones_like(background_map) * background_map_mean
homogeneous_medium = Medium(domain=domain,
        sound_speed=FourierSeries(np.expand_dims(c0 * np.ones(N) * homogeneous_background_map, -1), domain), 
        density=FourierSeries(np.expand_dims(rho0 * np.ones(N) * homogeneous_background_map, -1), domain),
        pml_size=pml[0])


# In[383]:


ext = [0, N[0]*dx[0], N[1]*dx[1], 0]
plt.scatter(element_positions[1]*dx[1], element_positions[0]*dx[0],
            c='r', marker='o', s=5, label='transducer element')
plt.imshow(sound_speed.params, cmap='gray', extent=ext)
plt.colorbar(label='Speed of sound [m/s]')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.legend(prop={'size': 7})  # Decreased the size of the legend
plt.gca().invert_yaxis()
plt.show()


# ## Time

# In[384]:


from jwave.geometry import TimeAxis

time_axis = TimeAxis.from_medium(medium, cfl=0.3)

time_axis.dt


# ## Source

# In[350]:


from jwave.signal_processing import gaussian_window

t = jnp.arange(0, time_axis.t_end, time_axis.dt)
s = transducer_magnitude * jnp.sin(2 * jnp.pi * transducer_frequency * t)

variance = 1/transducer_frequency
mean = 3*variance
ss = []
delay_s = 1
for i in range(element_positions.shape[1]):
    if delay_s < 0:
        ss.append(gaussian_window(s, t, mean + (i-64) * delay_s * time_axis.dt, variance))
    elif delay_s > 0:
        ss.append(gaussian_window(s, t, mean + i * delay_s * time_axis.dt, variance))
    else:
        ss.append(gaussian_window(s, t, mean, variance))

plt.plot(ss[-1])
plt.xlabel('Time point')
plt.ylabel('Amplitude [Pa]')
plt.show()


# In[351]:


from jwave.geometry import Sources

sources = Sources(
    positions=tuple(map(tuple, element_positions)),
    signals=jnp.vstack([ss[i] for i in range(element_positions.shape[1])]),
    dt=time_axis.dt,
    domain=domain,
)


# # Run simulation

# In[385]:


from jwave.acoustics import simulate_wave_propagation

@jit
def compiled_simulator(sources):
    pressure = simulate_wave_propagation(medium, time_axis, sources=sources)
    return pressure

@jit
def compiled_homogeneous_simulator(sources):
    pressure = simulate_wave_propagation(homogeneous_medium, time_axis, sources=sources)
    return pressure


# In[353]:


pressure = compiled_simulator(sources)
homogeneous_pressure = compiled_homogeneous_simulator(sources)


# In[379]:


from jwave.utils import show_field

t_idx = 300
show_field(pressure[t_idx])
plt.title(f"Pressure field at t={time_axis.to_array()[t_idx]} seconds")
plt.show()


# In[355]:


data = np.squeeze(pressure.params[:, element_positions[0], element_positions[1]])
homogeneous_data = np.squeeze(homogeneous_pressure.params[:, element_positions[0], element_positions[1]])


# In[356]:


plt.imshow(data, aspect='auto', cmap='seismic')
plt.xlabel('Transducer elements')
plt.ylabel('Time point')
plt.show()


# In[357]:


plt.imshow(homogeneous_data, aspect='auto', cmap='seismic')
plt.xlabel('Transducer elements')
plt.ylabel('Time point')
plt.show()


# In[358]:


# input_signal = np.stack([s for _ in range(nelements)], axis=1)
output_data = data - homogeneous_data
plt.imshow(output_data, aspect='auto', cmap='seismic')
plt.xlabel('Transducer elements')
plt.ylabel('Time point')
plt.show()


# In[ ]:





# In[ ]:





# # Beamforming

# In[ ]:





# In[359]:


slope = delay_s * time_axis.dt * c0 / dx[0].item()
oblique_factor = (slope*slope+1)**0.5

s_stack_t = torch.from_numpy(np.asarray(jnp.vstack([ss[i] for i in range(element_positions.shape[1])]))).to("cuda:0")
s_stack_t /= torch.max(s_stack_t).item()

def compute_time_delays_for_point(x1: np.ndarray, x: int, delta_y: int, c: float = c0):
    scaled_y = delta_y * dx[1]
    scaled_dx = (x1 - x) * dx[0]
    return np.round((scaled_y * oblique_factor + np.sqrt(scaled_y*scaled_y + scaled_dx * scaled_dx)) / (c  * time_axis.dt)).astype(int).tolist()

element_y = (N[1] - transducer_depth)

s_t = torch.from_numpy(np.asarray(s)).to("cuda:0")
s_t /= torch.max(s_t).item()
output_data_t = torch.from_numpy(np.asarray(output_data)).to("cuda:0")
element_positions_t = torch.from_numpy(element_positions[0]).to("cuda:0")
def compute_torch_signal(pt_x, pt_y):
    delta_y = abs(element_y - pt_y)
    delays = compute_time_delays_for_point(element_positions[0], pt_x, delta_y)
    signal = torch.tensor([0.0]).to("cuda:0")
    slanted_x_coord = int(pt_x - slope * delta_y)
    if slanted_x_coord > transducer_x_start + nelements - 1 or slanted_x_coord < transducer_x_start:
        return np.array([0.0])
    for i in range(len(delays)):
        if abs(element_positions[0][i] - (pt_x - slope * delta_y)) < delta_y:
            delta = delays[i]
            signal += torch.dot(s_stack_t[slanted_x_coord - transducer_x_start, :-delta], output_data_t[delta:, i])
    return signal * time_axis.dt


# In[360]:


# res is the old results
# res_oblique_minus_1 is one slant of results
def get_results():
    results = np.array([[compute_torch_signal(i, j).item() for j in range(0, 128)] for i in range(0, 128)])
    return results
res_oblique_plus_1 = get_results()


# In[278]:


output_data_t.shape


# In[374]:


from kwave.utils.filters import gaussian_filter
from kwave.reconstruction.beamform import envelope_detection

def postprocess_result(orig_res):
    result = np.copy(orig_res)
    for i in range(result.shape[0]):
        result[i, :] = gaussian_filter(result[i, :], 1/dx[0], transducer_frequency, 100.0)
    for i in range(result.shape[0]):
        result[i, :] = envelope_detection(result[i, :])
    # Plotting the heat map
    plt.imshow(result, cmap='viridis', interpolation='nearest')

    # Adding a color bar to show the scale
    plt.colorbar()

    # Display the heat map
    plt.show()
    return result

# postprocess_result(res_oblique_plus_1 + res_oblique_minus_1 + res)
a=postprocess_result(res_oblique_plus_1)
b=postprocess_result(res_oblique_minus_1)
c=postprocess_result(res)
plt.imshow((a+b+c)/3, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.show()


# In[ ]:





# In[375]:





# In[386]:


import time
all_results = []
for simulation_num in range(0, 11):
    delay_s = (simulation_num - 5)/5
    t = jnp.arange(0, time_axis.t_end, time_axis.dt)
    s = transducer_magnitude * jnp.sin(2 * jnp.pi * transducer_frequency * t)

    variance = 1/transducer_frequency
    mean = 3*variance
    ss = []
    for i in range(element_positions.shape[1]):
        if delay_s < 0:
            ss.append(gaussian_window(s, t, mean + (i-64) * delay_s * time_axis.dt, variance))
        elif delay_s > 0:
            ss.append(gaussian_window(s, t, mean + i * delay_s * time_axis.dt, variance))
        else:
            ss.append(gaussian_window(s, t, mean, variance))
    sources = Sources(
        positions=tuple(map(tuple, element_positions)),
        signals=jnp.vstack([ss[i] for i in range(element_positions.shape[1])]),
        dt=time_axis.dt,
        domain=domain,
    )
    pressure = compiled_simulator(sources)
    homogeneous_pressure = compiled_homogeneous_simulator(sources)
    data = np.squeeze(pressure.params[:, element_positions[0], element_positions[1]])
    homogeneous_data = np.squeeze(homogeneous_pressure.params[:, element_positions[0], element_positions[1]])
    output_data = data - homogeneous_data
    slope = delay_s * time_axis.dt * c0 / dx[0].item()
    oblique_factor = (slope*slope+1)**0.5

    s_stack_t = torch.from_numpy(np.asarray(jnp.vstack([ss[i] for i in range(element_positions.shape[1])]))).to("cuda:0")
    s_stack_t /= torch.max(s_stack_t).item()

    def compute_time_delays_for_point(x1: np.ndarray, x: int, delta_y: int, c: float = c0):
        scaled_y = delta_y * dx[1]
        scaled_dx = (x1 - x) * dx[0]
        return np.round((scaled_y * oblique_factor + np.sqrt(scaled_y*scaled_y + scaled_dx * scaled_dx)) / (c  * time_axis.dt)).astype(int).tolist()

    element_y = (N[1] - transducer_depth)

    output_data_t = torch.from_numpy(np.asarray(output_data)).to("cuda:0")
    element_positions_t = torch.from_numpy(element_positions[0]).to("cuda:0")
    def compute_torch_signal(pt_x, pt_y):
        delta_y = abs(element_y - pt_y)
        delays = compute_time_delays_for_point(element_positions[0], pt_x, delta_y)
        signal = torch.tensor([0.0]).to("cuda:0")
        slanted_x_coord = int(pt_x - slope * delta_y)
        if slanted_x_coord > transducer_x_start + nelements - 1 or slanted_x_coord < transducer_x_start:
            return np.array([0.0])
        for i in range(len(delays)):
            if abs(element_positions[0][i] - (pt_x - slope * delta_y)) < delta_y:
                delta = delays[i]
                signal += torch.dot(s_stack_t[slanted_x_coord - transducer_x_start, :-delta], output_data_t[delta:, i])
        return signal * time_axis.dt
    def get_results():
        results = np.array([[compute_torch_signal(i, j).item() for j in range(10, 108)] for i in range(10, 118)])
        return results
    all_results.append(get_results())
    print(f"done with {simulation_num} at time {time.time()}")
def postprocess_result(orig_res):
    result = np.copy(orig_res)
    for i in range(result.shape[0]):
        result[i, :] = gaussian_filter(result[i, :], 1/dx[0], transducer_frequency, 100.0)
    for i in range(result.shape[0]):
        result[i, :] = envelope_detection(result[i, :])
    # Plotting the heat map
    plt.imshow(result, cmap='viridis', interpolation='nearest')

    # Adding a color bar to show the scale
    plt.colorbar()

    # Display the heat map
    plt.show()
    return result
for result in all_results:
    postprocess_result(result)


# In[397]:


x= postprocess_result(all_results[0])
for i in range(1,11):
    x += postprocess_result(all_results[i])
plt.imshow(x, cmap='viridis', interpolation='nearest')
plt.colorbar()


# In[395]:


np.sum(all_results[0] - all_results[1])

