#!/usr/bin/env python3 

import numpy as np
import matplotlib.pyplot as plt

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

def add_landmarks(map_array, list_of_landmarks):
    # Get the size of the map
    x, y = map_array.shape

    # Create a list of random coordinates
    coordinates = list_of_landmarks

    # Add the landmarks to the map
    for i, coordinate in enumerate(coordinates):
        map_array[coordinate[0], coordinate[1]] = i+2

    return map_array

map = generate_map(8*4,7*12, borders=False)
np.save('map/map_1.npy', map)

list_of_landmarks = [[0, 0], [5, 0], [28, 0], [36, 0], [41, 0], [83, 0],
                     [0, 4], [83, 4],
                     [10, 8], [15, 8], [20, 8], [25, 8], [30, 8], [35, 8], [42, 8], [48, 8], [56, 8],
                     [0, 16],
                     [42, 18],
                     [6,20], 
                     [36,22], [48, 22],
                     [0, 24], [12, 24],
                     [42, 26], [83, 27],
                     [6, 28], [83, 29],
                     [10, 31], [24, 31], [83, 31]]
list_of_landmarks = [[y, x] for [x, y] in list_of_landmarks] 

map_landmarks = add_landmarks(map, list_of_landmarks)         
np.save('map/map_landmarks.npy', map_landmarks)
plt.imshow(map_landmarks)
plt.show()