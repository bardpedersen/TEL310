#!/usr/bin/env/python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as mp

# Load the image
image = Image.open('map1.jpg')  

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a NumPy array
image_array = np.array(gray_image)

# Threshold the array to set black pixels to 1 and white pixels to 0
binary_array = (image_array < 55).astype(np.uint8)

# erode the image
binary_array = mp.erosion(binary_array, mp.square(7))

# dilate the image
binary_array = mp.dilation(binary_array, mp.square(25))

# erode the image
binary_array = mp.erosion(binary_array, mp.square(18))

# Save the binary array as an NPY file
np.save('binary_image.npy', binary_array)

# Load the binary array
binary_array = np.load('binary_image.npy')

# Plot image
plt.imshow(binary_array, cmap='Greys')
plt.show()

