#!/usr/bin/env/python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as mp

# Load the image
image = Image.open('map_cats.jpg')  

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a NumPy array
image_array = np.array(gray_image)

# Threshold the array to set black pixels to 1 and white pixels to 0
binary_array = (image_array < 200).astype(np.uint8)


# Save the binary array as an NPY file
np.save('binary_image_cats.npy', binary_array)

# Load the binary array
binary_array = np.load('binary_image_cats.npy')

# Plot image
plt.imshow(binary_array, cmap='Greys')
plt.show()

