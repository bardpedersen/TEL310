#!/usr/bin/env python3 

import numpy as np

def generate_map(x, y, borders=True):
    # Create a zero-filled numpy array of size x by y
    map_array = np.zeros((x, y))

    if borders:
        # Set the borders to 1
        map_array[0, :] = 1  # Top border
        map_array[:, 0] = 1  # Left border
        map_array[-1, :] = 1  # Bottom border
        map_array[:, -1] = 1  # Right border

    return map_array

np.save('map/map_1.npy', generate_map(17, 17, borders=False))