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
        map_array[coordinate[0], coordinate[1]] = coordinate[2]

    return map_array

map = generate_map(8*4,7*12, borders=False)
np.save('map_1.npy', map)

list_of_landmarks = [[0, 0, 1], [5, 0, 2], [28, 0, 3], [36, 0, 4], [41, 0, 5], [83, 0, 6],
                     [0, 4, 7], [83, 4, 8],
                     [10, 8, 9] , [15, 8, 10], [20, 8, 11], [25, 8, 12], [30, 8, 13], [35, 8, 14], [42, 8, 15], [48, 8, 16], [56, 8, 17],
                     [0, 16, 18],
                     [42, 18, 19],
                     [6, 20, 20], 
                     [36, 22, 21], [48, 22, 22],
                     [0, 24, 23], [12, 24, 24],
                     [42, 26, 25], [83, 27, 26],
                     [6, 28, 27], [83, 29, 28],
                     [10, 31, 29], [24, 31, 30], [83, 31, 31]]

list_of_landmarks = [[y, x, z] for [x, y, z] in list_of_landmarks]  # Swap x and y

map_landmarks = add_landmarks(map, list_of_landmarks)         
np.save('map_landmarks.npy', map_landmarks)
plt.imsave('map_landmarks.png', map_landmarks)
plt.close()