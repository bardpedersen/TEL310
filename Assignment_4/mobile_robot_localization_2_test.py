#!/usr/bin/env python3 

import numpy as np
import sensor_model as sensor_model
import mobile_robot_localization_2 as mrl2
import matplotlib.pyplot as plt
import os
from PIL import Image


def Grid_localization_test():
    m = np.load('map/binary_image_cats.npy')
    pkt_1 = np.ones((15,15)) / (15*15)
    ut = [10, 0]
    zt = np.array([218.0, 182, 500, 303.0, 355.0, 496.0, 182, 206]) # exact values 
    noise = np.random.normal(0, 2, size=len(zt)) 
    zt += noise # add noise to the values
    zt[2] = 500 # to not get larger than 500

    z_hit = 0.8157123986145881
    z_short = 0.00235666025958796
    z_max = 0.16552184295092348
    z_rand =  0.01640909817490046
    sigma_hit = 1.9665518618953464
    lambda_short = 0.0029480342354130016
    theta = np.array([z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short])
    z_maxlen = 500
    delta_t = 1
    alpha = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

    p_kt = mrl2.Grid_localization(pkt_1, ut, zt, m, delta_t, alpha, theta, z_maxlen)
    p_kt = np.transpose(np.array(p_kt))
    print(np.sum(p_kt))
    plt.imshow(p_kt)
    plt.show()

def generate_particles(number_of_particles, m):
    list_of_particles = []
    free_spaces = np.argwhere(m == 0)  # Get indices of free spaces
    for _ in range(number_of_particles):
        random_index = free_spaces[np.random.choice(free_spaces.shape[0])]  # Randomly select one index
        random_theta = np.random.uniform(0, 2*np.pi)  # Generate a random theta
        list_of_particles.append([[random_index[1], random_index[0], random_theta], 1/number_of_particles])

    return np.array(list_of_particles, dtype=object)

def find_z_values(xt_list, m):
    zt = []
    for xt in xt_list:
        z_maxlen = 500
        zt_start = np.pi/4
        zt_end = -np.pi/4
        number_of_readings = 8
        zt1 = [sensor_model.ray_casting(xt, m, theta, z_maxlen, transpose=True) for theta in np.linspace(zt_start, zt_end, number_of_readings)]
        zt1 = np.array(zt1) 
        noise = np.random.normal(0, 2, size=len(zt1))  # Generate noise
        zt1 += noise  # Add noise to zt
        zt1 = np.minimum(zt1, z_maxlen)  # Limit maximum value
        zt.append(zt1)

    return zt

def create_MCL_GIF():
    
    filenames = MCL_test()
    # Convert the individual plots to a GIF
    with Image.open(filenames[0]) as im:
        im.save("images/animation_MCL.gif", save_all=True, append_images=[Image.open(f) for f in filenames[1:]], duration=100, loop=0)

    # Remove the individual plot files
    for f in filenames:
        os.remove(f)


def MCL_test():
    list_of_plots = []
    m = np.load('map/binary_image_cats.npy')
    number_of_particles = 50
    Xt_1 = generate_particles(number_of_particles, m)
    xt_list = [[200, 500, np.pi/2], [200, 450, np.pi/2], [200, 400, np.pi/2], [200, 400, np.pi], [150, 400, np.pi], [100, 400, np.pi]]
    ut = [[50, 0], [50, 0], [50, 0], [0, np.pi/2], [50, 0], [50, 0]]
    zt = find_z_values(xt_list, m)

    for i, utt in enumerate(ut):
        Xt = mrl2.MCL(Xt_1, utt, zt[i], m)
        positions = np.array([sublist[0] for sublist in Xt])
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]
        weights = [sublist[1] for sublist in Xt]
        positions1 = np.array([sublist[0] for sublist in Xt_1])
        x_positions1 = positions1[:, 0]
        y_positions1 = positions1[:, 1]

        plt.title("Sample landmark model")
        plt.imshow(np.load('map/binary_image.npy'), cmap='Greys')
        plt.scatter(x_positions, y_positions, c='b', marker='.')
        plt.scatter(x_positions1, y_positions1, c='r', marker='o')
        plt.scatter(xt_list[i][0], xt_list[i][1], c='g', marker='o')
        plt.savefig(f"images/step_{i}.png")
        list_of_plots.append(f"images/step_{i}.png")
        plt.close()
        Xt_1 = Xt

    return list_of_plots

if __name__ == "__main__":
    #Grid_localization_test()
    #MCL_test()
    create_MCL_GIF()