import numpy as np


def match_time_step(time_steps, obs_time_step):
    # Warning: assumes all time steps are unique and that there will be only one index
    # returned by np.argmin
    time_steps = np.array(time_steps)
    time_steps_ref = obs_time_step * np.ones_like(time_steps)
    time_diff = np.abs(time_steps_ref - time_steps)
    return np.argmin(time_diff)
