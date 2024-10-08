import numpy as np
from scipy.signal.windows import hann
import jax.numpy as jnp
from jax import jit
from jwave import FourierSeries, FiniteDifferences
from jwave.geometry import Domain, Medium, Sources, Sensors
from jwave.acoustics import simulate_wave_propagation
from jwave.signal_processing import gaussian_window
from kwave.utils.signals import tone_burst


def get_domain(N, dx):
    """
    Get the jwave spatial domain.

    Parameters
    ----------
    N : np.ndarray
        Number of grid points in each dimension.
    dx : np.ndarray
        Grid spacing in each dimension in meters.

    Returns
    -------
    domain : jwave.geometry.Domain
        Spatial domain.
    """
    domain = Domain((N[0],N[1]), (dx[0], dx[1]))
    return domain


def get_background(N, mean, std, random_seed=28):
    """
    Get a random distribution of background scatterers.

    Parameters
    ----------
    N : np.ndarray
        Number of grid points in each dimension.
    mean : float
        Mean of the background map.
    std : float
        Standard deviation of the background map.
    random_seed : int
        Seed for the random number generator.

    Returns
    -------
    background_map : np.ndarray
        Map of background scatterers.
    """
    np.random.seed(random_seed)
    background_map = mean + std * np.random.randn(N[0], N[1])
    return background_map


def get_homogeneous_medium(domain, c0=1500, rho0=1000, pml_size=20, 
                           background_mean=1, background_std=0.008, 
                           background_seed=28):
    """
    Get a homogeneous acoustic medium.

    Parameters
    ----------
    domain : jwave.geometry.Domain
        Spatial domain.
    c0 : float
        Reference speed of sound in m/s.
    rho0 : float
        Reference density in kg/m^3.
    pml_size : int
        Size of the PML in grid points.
    background_mean : float
        Mean of the background map.
    background_std : float
        Standard deviation of the background map.
    background_seed : int
        Seed for the random number generator.

    Returns
    -------
    medium : jwave.medium.Medium
        Medium.
    """
    sound_speed = c0 * np.ones(domain.N)
    density = rho0 * np.ones(domain.N)

    # add background noise
    background_map = get_background(domain.N, mean=background_mean, 
                                    std=background_std, 
                                    random_seed=background_seed)
    sound_speed = sound_speed * background_map
    density = density * background_map

    return sound_speed, density


def get_scatterers(N, positions, radius, contrast):
    """
    Get scatterers.

    Parameters
    ----------
    N : np.ndarray
        Number of grid points in each dimension.
    positions : np.ndarray
        Positions of the scatterers in grid points.
    radius : int
        Radius of scatterers in grid points.    
    contrast : float
        Contrast of scatterers.

    Returns
    -------
    scatterer_map : np.ndarray
        Map of scatterers.
    """
    scatterer_map = np.zeros(N)
    x, y = np.ogrid[:N[0], :N[1]]
    for position in positions:
        scatterer_map[(x - position[0])**2 + (y - position[1])**2 <= (radius)**2] = 1
    return scatterer_map


def get_point_medium(domain, scatterer_positions, scatterer_radius=2, scatterer_contrast=1.1,
                     c0=1500, rho0=1000, pml_size=20,
                     background_mean=1, background_std=0.008, background_seed=28):
    """
    Get an acoustic medium with defined point scatterers.

    Parameters
    ----------
    domain : jwave.geometry.Domain
        Spatial domain.
    scatterer_positions : np.ndarray
        Positions of the scatterers in grid points.
    scatterer_radius : int
        Radius of the scatterer in grid points.
    scatterer_contrast : float
        Contrast of the scatterer.
    c0 : float
        Reference speed of sound in m/s.
    rho0 : float
        Reference density in kg/m^3.
    pml_size : int
        Size of the PML in grid points.
    background_mean : float
        Mean of the background map.
    background_std : float
        Standard deviation of the background map.
    background_seed : int
        Seed for the random number generator.

    Returns
    -------
    medium : jwave.medium.Medium
        Medium.
    """
    sound_speed = c0 * np.ones(domain.N)
    density = rho0 * np.ones(domain.N)

    # add background noise
    background_map = get_background(domain.N, mean=background_mean, 
                                    std=background_std, 
                                    random_seed=background_seed)
    sound_speed = sound_speed * background_map
    density = density * background_map

    # add scatterers
    scatterer_map = get_scatterers(domain.N, scatterer_positions, scatterer_radius, scatterer_contrast)
    sound_speed[scatterer_map == 1] = c0*scatterer_contrast
    density[scatterer_map == 1] = rho0*scatterer_contrast

    return sound_speed, density

def get_skull_medium(domain, skull_slice, 
                           scatterer_positions=None, scatterer_radius=2, scatterer_contrast=1.1,
                           c0=1500, rho0=1000, pml_size=20,
                           background_mean=1, background_std=0.008, background_seed=28):
    """
    Get an acoustic medium with a skull and optionally, defined point scatterers.

    Parameters
    ----------
    domain : jwave.geometry.Domain
        Spatial domain.
    skull_slice : np.ndarray
        Skull slice.
    scatterer_positions : np.ndarray
        Positions of the scatterers in grid points.
    scatterer_radius : int
        Radius of the scatterer in grid points.
    scatterer_contrast : float
        Contrast of the scatterer.
    c0 : float
        Reference speed of sound in m/s.
    rho0 : float
        Reference density in kg/m^3.
    pml_size : int
        Size of the PML in grid points.
    background_mean : float
        Mean of the background map.
    background_std : float
        Standard deviation of the background map.
    background_seed : int
        Seed for the random number generator.

    Returns
    -------
    medium : jwave.medium.Medium
        Medium.
    """
    sound_speed = c0 * np.ones(domain.N)
    density = rho0 * np.ones(domain.N)

    # add background noise
    background_map = get_background(domain.N, mean=background_mean, 
                                    std=background_std, 
                                    random_seed=background_seed)
    sound_speed = sound_speed * background_map
    density = density * background_map

    # add scatterers
    if scatterer_positions is not None:
        scatterer_map = get_scatterers(domain.N, scatterer_positions, scatterer_radius, scatterer_contrast)
        sound_speed[scatterer_map == 1] = c0*scatterer_contrast
        density[scatterer_map == 1] = rho0*scatterer_contrast

    # add skull
    skull_mask = skull_slice > 20000
    sound_speed[skull_mask] = 2700
    density[skull_mask] = 1800

    return sound_speed, density


def get_plane_wave_excitation(domain, time_axis, magnitude, frequency, pitch, positions, angle=0, c0=1550, hann_window=False, tone=False):
    """
    Get a plane wave excitation from a linear probe.
    
    Parameters
    ----------
    domain : jwave.geometry.Domain
        Spatial domain.
    time_axis : jwave.geometry.TimeAxis
        Time axis.
    magnitude : float
        Magnitude of the excitation.
    frequency : float
        Frequency of the excitation.
    pitch : float
        Pitch of the transducer elements in meters.
    positions : np.ndarray
        Positions of the sources in grid points.
    angle : float
        Angle of the plane wave in radians relative to the probe's normal.
    c0 : float
        Reference speed of sound in m

    Returns
    -------
    sources : jwave.geometry.Sources
        Sources.
    signal : np.ndarray
        Signal emitted at each transmitter.
    carrier_signal : np.ndarray
        Carrier signal at the center frequency of the probe.
    """
    # TODO(vincent): this doesn't work for signal delays other than 0 and different transducer spacings
    nelements = positions.shape[1]
    t = jnp.arange(0, time_axis.t_end, time_axis.dt)
    carrier_signal = magnitude * jnp.sin(2 * jnp.pi * frequency * t)
    variance = 1/frequency
    mean = 3*variance
    signal = []
    signal_delay = pitch * np.sin(angle) / c0

    if tone:
        ncycles = 2.5
        signal_w = tone_burst(1/time_axis.dt, frequency, ncycles, signal_length = int(time_axis.Nt))
        for i in range(nelements):
            if angle < 0:
                signal.append(np.roll(signal_w, -int(i * signal_delay / time_axis.dt)))
            elif angle > 0:
                signal.append(np.roll(signal_w, -int((i-nelements) * signal_delay / time_axis.dt)))
            else:
                signal.append(signal_w)
    elif hann_window:
        ncycles = 2.5
        cycle_duration = ncycles / frequency
        nsamples_per_cycle = int(cycle_duration / time_axis.dt)
        window = hann(nsamples_per_cycle)
        signal_w = carrier_signal[:nsamples_per_cycle] * window
        signal_w = np.pad(signal_w, (0, int(time_axis.Nt) - nsamples_per_cycle), mode='constant')

        for i in range(nelements):
            # delay = int(i * signal_delay / time_axis.dt)
            if angle < 0:
                signal.append(np.roll(signal_w, -int(i * signal_delay / time_axis.dt)))
            elif angle > 0:
                signal.append(np.roll(signal_w, -int((i-nelements) * signal_delay / time_axis.dt)))
            else:
                signal.append(signal_w)
    else:
        for i in range(positions.shape[1]):
            if angle < 0:
                signal.append(gaussian_window(carrier_signal, t, mean + (i-nelements) * signal_delay, variance))
            elif angle > 0:
                signal.append(gaussian_window(carrier_signal, t, mean + i * signal_delay, variance))
            else:
                signal.append(gaussian_window(carrier_signal, t, mean, variance))
        
    sources = Sources(
        positions=tuple(map(tuple, positions)),
        signals=jnp.vstack([signal[i] for i in range(positions.shape[1])]),
        dt=time_axis.dt,
        domain=domain,
    )

    return sources, signal, carrier_signal


def get_data(sound_speed, density, domain, time_axis, sources, sensor_positions, pml_size=20):
    """
    Get the ultrasound pressure data.

    Parameters
    ----------
    sound_speed : np.ndarray
        Sound speed in m/s.
    density : np.ndarray
        Density in kg/m^3.
    domain : jwave.geometry.Domain
        Spatial domain.
    time_axis : jwave.geometry.TimeAxis
        Time axis.
    sources : jwave.geometry.Sources
        Sources.
    sensor_positions : np.ndarray
        Positions of the sensors in grid points.

    Returns
    -------
    pressure : jwave.geometry.Pressure
        Pressure.
    data : np.ndarray
        Pressure data.
    """
    # get medium
    sound_speed = FourierSeries(jnp.expand_dims(sound_speed, -1), domain)
    # sound_speed = FiniteDifferences(jnp.expand_dims(sound_speed, -1), domain)
    density = FourierSeries(jnp.expand_dims(density, -1), domain)
    # density = FiniteDifferences(jnp.expand_dims(density, -1), domain)
    medium = Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=pml_size)

    # run simulation
    @jit
    def compiled_simulator(sources):
        pressure = simulate_wave_propagation(medium, time_axis, sources=sources)
        return pressure
    pressure = compiled_simulator(sources)

    # get pressure data
    data = jnp.squeeze(pressure.params[:, sensor_positions[0], sensor_positions[1]])
    
    return pressure, data

@jit
def get_data_only(sound_speed, density, domain, time_axis, sources, sensor_positions, pml_size=20):

    # get medium
    sound_speed = FourierSeries(jnp.expand_dims(sound_speed, -1), domain)
    density = FourierSeries(jnp.expand_dims(density, -1), domain)
    medium = Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=pml_size)
    sensors = Sensors(sensor_positions)

    # run simulation
    data = simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors)

    return data[..., 0]