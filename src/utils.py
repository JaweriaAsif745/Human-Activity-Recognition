import numpy as np

def sample_frame_indices(num_frames_total, num_frames_sample):
    if num_frames_total <= num_frames_sample:
        return np.linspace(0, num_frames_total - 1, num_frames_sample, dtype=int)
    else:
        interval = num_frames_total / num_frames_sample
        return np.array([int(i * interval) for i in range(num_frames_sample)])