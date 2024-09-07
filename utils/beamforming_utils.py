import numpy as np
import jax.numpy as jnp


def get_receive_beamforming(domain, time_axis, positions, output_data, signal, carrier_signal, signal_delay, c0=1500):
    """
    Get the receive beamforming data.

    Parameters
    ----------
    domain : jwave.geometry.Domain
        Spatial domain.
    time_axis : jwave.geometry.TimeAxis
        Time axis.
    positions : np.ndarray
        Transducer element positions.
    output_data : np.ndarray
        Pressure data at the receiver positions.
    signal : np.ndarray
        Emitted source signal at transmitter positions.
    carrier_signal : np.ndarray
        Carrier source signal.
    signal_delay : int
        Signal delay in timepoints.
    c0 : float
        Reference sound speed in meters/second.

    Returns
    -------
    receive : np.ndarray
        Receive beamforming data.
    """

    slope = signal_delay * time_axis.dt * c0 / domain.dx[0].item()
    oblique_factor = (slope*slope+1)**0.5
    s_stack_t = np.asarray(jnp.vstack([signal[i] for i in range(positions.shape[1])]))
    s_stack_t = s_stack_t / np.max(s_stack_t)

    def compute_time_delays_for_point(x1: np.ndarray, x: int, delta_y: int, c: float = c0):
        scaled_y = delta_y * domain.dx[1]
        scaled_dx = (x1 - x) * domain.dx[0]
        return np.round((scaled_y * oblique_factor + np.sqrt(scaled_y*scaled_y + scaled_dx * scaled_dx)) / (c  * time_axis.dt)).astype(int).tolist()

    output_data_t = np.asarray(output_data)

    transducer_x_start = positions[0][0]
    transducer_x_end = positions[0][-1]
    transducer_y = positions[1][0]
    def compute_signal(pt_x, pt_y):
        delta_y = abs(transducer_y - pt_y)
        delays = compute_time_delays_for_point(positions[0], pt_x, delta_y)
        signal = np.zeros_like(output_data_t[:, 0])
        slanted_x_coord = int(pt_x - slope * delta_y)
        if slanted_x_coord > transducer_x_end or slanted_x_coord < transducer_x_start:
            return 0.0
        for i, delta in enumerate(delays):
            if abs(positions[0][i] - (pt_x - slope * delta_y)) < delta_y * abs(positions[0][1] - positions[0][0]):
                signal[:-delta] += output_data_t[delta:, i]
        return (np.dot(signal, s_stack_t[(slanted_x_coord - transducer_x_start) // abs(positions[0][1] - positions[0][0])]) * time_axis.dt).item()

    receive = np.array([[compute_signal(i, j) for j in range(0, domain.N[1])] for i in range(0, domain.N[0])])
    return receive

def get_receive_beamforming_medium_specific(domain, medium, time_axis, positions, output_data, signal, carrier_signal, signal_delay, c0=1500):
    """
    Get the receive beamforming data.

    Parameters
    ----------
    domain : jwave.geometry.Domain
        Spatial domain.
    time_axis : jwave.geometry.TimeAxis
        Time axis.
    positions : np.ndarray
        Transducer element positions.
    output_data : np.ndarray
        Pressure data at the receiver positions.
    signal : np.ndarray
        Emitted source signal at transmitter positions.
    carrier_signal : np.ndarray
        Carrier source signal.
    signal_delay : int
        Signal delay in timepoints.
    c0 : float
        Reference sound speed in meters/second.

    Returns
    -------
    receive : np.ndarray
        Receive beamforming data.
    """

    slope = signal_delay * time_axis.dt * c0 / domain.dx[0].item()
    # slope is in pixel space
    s_stack_t = np.asarray(jnp.vstack([signal[i] for i in range(positions.shape[1])]))
    s_stack_t = s_stack_t / np.max(s_stack_t)

    output_data_t = np.asarray(output_data)

    transducer_x_start = positions[0][0]
    transducer_x_end = positions[0][-1]
    transducer_y = positions[1][0]

    transducer_spacing = abs(positions[0][1] - positions[0][0])

    all_delays = np.zeros([domain.N[0], domain.N[1], positions.shape[1]])
    
    # compute a 3D array of the estimated delays from every point in the grid to every transducer point
    x_range = np.arange(0, domain.N[0])
    pos_arr = [[i for i in range(positions.shape[1])]]*domain.N[0]
    for j in range(transducer_y - 1, -1, -1):

        prev_row = (positions[0][None, :] + x_range[:, None] * (transducer_y - j - 1)) / (transducer_y - j)
        prev_row_floor = np.floor(prev_row).astype(int)
        frac = prev_row - prev_row_floor
        prev_delays = (1-frac) * all_delays[prev_row_floor, j+1, pos_arr] + frac * all_delays[prev_row_floor+1, j+1, pos_arr]
        ratios = (positions[0][None, :] - x_range[:, None]) / (transducer_y - j)
        new_delays = np.sqrt(np.square(ratios) + 1) * domain.dx[0] / (medium.sound_speed.params[:, j, 0] * time_axis.dt)[:,None]
        all_delays[:, j] = prev_delays + new_delays


    def compute_time_delays_for_point(x: int, transducer_y: int, pt_y: int):
        first_delay = all_delays[x,pt_y,int((x - (transducer_y - pt_y) * slope - transducer_x_end) // transducer_spacing)]
        second_delay = all_delays[x, pt_y, :]
        return np.round(first_delay + second_delay).astype(int).tolist()

    def compute_signal(pt_x, pt_y):
        if transducer_y <= pt_y:
            return 0.0
        delta_y = transducer_y - pt_y
        signal = np.zeros_like(output_data_t[:, 0])
        slanted_x_coord = int(pt_x - slope * delta_y)
        if slanted_x_coord > transducer_x_end or slanted_x_coord < transducer_x_start:
            return 0.0
        delays = compute_time_delays_for_point(pt_x, transducer_y, pt_y)
        for i, delta in enumerate(delays):
            if delta > 0 and abs(positions[0][i] - (pt_x - slope * delta_y)) < delta_y * transducer_spacing:
                signal[:-delta] += output_data_t[delta:, i]
        return (np.dot(signal, s_stack_t[(slanted_x_coord - transducer_x_start) // transducer_spacing]) * time_axis.dt).item()

    receive = np.array([[compute_signal(i, j) for j in range(0, domain.N[1])] for i in range(0, domain.N[0])])
    return receive