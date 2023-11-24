#!/usr/bin/env python3 

import numpy as np
import sensor_model as sensor_model
import mobile_robot_localization_2 as mrl2
import matplotlib.pyplot as plt
import os
from PIL import Image



def Grid_localization_test():
    m = np.load('map/binary_image_cats.npy')
    pkt_1 = np.ones((10,10)) / (10*10)
    ut = [10, 0]
    xt = [[500, 50, np.pi/2]]
    zt = find_z_values(xt, m)

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

    p_kt = mrl2.Grid_localization(pkt_1, ut, zt[0], m, delta_t, alpha, theta, z_maxlen)
    p_kt = np.transpose(np.array(p_kt))
    extent = [0, m.shape[1], m.shape[0], 0]  # [left, right, bottom, top]

    plt.imshow(m, extent=extent, cmap='Greys')
    plt.scatter(xt[0][0], xt[0][1], c='g', marker='o')
    plt.imshow(p_kt, alpha=0.5, extent=extent, cmap='Greys')
    plt.savefig('images/grid_localization.png')
    plt.close()

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
        zt_start = 0
        zt_end = np.pi
        number_of_readings = 8
        zt1 = [sensor_model.ray_casting(xt, m, theta, z_maxlen, transpose=True) for theta in np.linspace(zt_start, zt_end, number_of_readings)]
        zt1 = np.array(zt1) 
        noise = np.random.normal(0, 2, size=len(zt1))  # Generate noise
        zt1 += noise  # Add noise to zt
        zt1 = np.minimum(zt1, z_maxlen)  # Limit maximum value
        zt.append(zt1)

    return zt

def find_z_values_1(xt_list, m):
    zt = []
    for xt in xt_list:
        z_maxlen = 500
        zt_start = 0
        zt_end = 0
        number_of_readings = 1
        zt1 = [sensor_model.ray_casting(xt, m, theta, z_maxlen, transpose=True) for theta in np.linspace(zt_start, zt_end, number_of_readings)]
        zt1 = np.array(zt1) 
        noise = np.random.normal(0, 2, size=len(zt1))  # Generate noise
        zt1 += noise  # Add noise to zt
        zt1 = np.minimum(zt1, z_maxlen)  # Limit maximum value
        zt.append(zt1)
    return zt

def create_GIF(function):
    # Convert the individual plots to a GIF
    with Image.open(function[0]) as im:
        im.save("images/animation_MCL.gif", save_all=True, append_images=[Image.open(f) for f in function[1:]], duration=1000, loop=0)

    # Remove the individual plot files
    for f in function:
        os.remove(f)


def MCL_test():
    list_of_plots = []
    m = np.load('map/binary_image_cats.npy')
    number_of_particles = 50
    Xt_1 = generate_particles(number_of_particles, m)
    xt_list = [[300, 300, np.pi/2], [300, 275, np.pi/2], [300, 250, np.pi/2], [300, 225, np.pi/2]]
    ut = [[25, 0], [25, 0], [25, 0], [25, 0]]
    zt = find_z_values(xt_list, m)

    positions1 = np.array([sublist[0] for sublist in Xt_1])
    x_positions1 = positions1[:, 0]
    y_positions1 = positions1[:, 1]
    
    plt.title("MCL")
    plt.imshow(m, cmap='Greys')
    plt.scatter(x_positions1, y_positions1, c='r', marker='.')
    plt.scatter(xt_list[0][0], xt_list[0][1], c='g', marker='o')
    plt.savefig(f"images/step_{0}.png")
    list_of_plots.append(f"images/step_{0}.png")
    plt.close()

    for i, utt in enumerate(ut):
        print(f"Step {i}")
        Xt = mrl2.MCL(Xt_1, utt, zt[i], m)
        positions = np.array([sublist[0] for sublist in Xt])
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]

        plt.title("MCL ")
        plt.imshow(m, cmap='Greys')
        plt.scatter(x_positions, y_positions, c='r', marker='.')
        plt.scatter(xt_list[i][0], xt_list[i][1], c='g', marker='o')
        plt.savefig(f"images/step_{i+1}.png")
        list_of_plots.append(f"images/step_{i+1}.png")
        plt.close()
        Xt_1 = Xt

    return list_of_plots


def Augmented_MCL_test():
    list_of_plots = []
    m = np.load('map/binary_image_cats.npy')
    number_of_particles = 50
    Xt_1 = generate_particles(number_of_particles, m)
    xt_list = [[400, 500, np.pi/2], [400, 450, np.pi/2], [400, 400, np.pi/2], [400, 350, np.pi/2], [400, 300, np.pi/2], [400, 250, np.pi/2]]
    ut = [[50, 0], [50, 0], [50, 0], [50, 0], [50, 0], [50, 0]]
    zt = find_z_values(xt_list, m)

    positions1 = np.array([sublist[0] for sublist in Xt_1])
    x_positions1 = positions1[:, 0]
    y_positions1 = positions1[:, 1]
    
    plt.title("Augmented_MCL")
    plt.imshow(m, cmap='Greys')
    plt.scatter(x_positions1, y_positions1, c='r', marker='.')
    plt.scatter(xt_list[0][0], xt_list[0][1], c='g', marker='o')
    plt.savefig(f"images/step_{0}.png")
    list_of_plots.append(f"images/step_{0}.png")
    plt.close()

    for i, utt in enumerate(ut):
        print(f"Step {i}")
        Xt = mrl2.Augmented_MCL(Xt_1, utt, zt[i], m)
        positions = np.array([sublist[0] for sublist in Xt])
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]
        weights = [sublist[1] for sublist in Xt]

        plt.title("Augmented_MCL ")
        plt.imshow(m, cmap='Greys')
        plt.scatter(x_positions, y_positions, c='r', marker='.')
        plt.scatter(xt_list[i][0], xt_list[i][1], c='g', marker='o')
        plt.savefig(f"images/step_{i+1}.png")
        list_of_plots.append(f"images/step_{i+1}.png")
        plt.close()
        Xt_1 = Xt

    return list_of_plots

def KLD_Sampling_MCL_test():
    list_of_plots = []
    m = np.load('map/binary_image_cats.npy')
    number_of_particles = 50
    Xt_1 = generate_particles(number_of_particles, m)
    xt_list = [[680, 500, np.pi/2], [680, 450, np.pi/2], [680, 400, np.pi/2], [680, 350, np.pi/2], [680, 300, np.pi/2], [680, 250, np.pi/2]]
    ut = [[50, 0], [50, 0], [50, 0], [50, 0], [50, 0], [50, 0]]
    zt = find_z_values(xt_list, m)
    epsilon = 0.1
    delta = 0.05

    positions1 = np.array([sublist[0] for sublist in Xt_1])
    x_positions1 = positions1[:, 0]
    y_positions1 = positions1[:, 1]
    
    plt.title("KLD_Sampling_MCL")
    plt.imshow(m, cmap='Greys')
    plt.scatter(x_positions1, y_positions1, c='r', marker='.')
    plt.scatter(xt_list[0][0], xt_list[0][1], c='g', marker='o')
    plt.savefig(f"images/step_{0}.png")
    list_of_plots.append(f"images/step_{0}.png")
    plt.close()

    for i, utt in enumerate(ut):
        print(f"Step {i}")
        Xt = mrl2.KLD_Sampling_MCL(Xt_1, utt, zt[i], m, epsilon, delta)
        positions = np.array([sublist[0] for sublist in Xt])
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]

        plt.title("KLD_Sampling_MCL ")
        plt.imshow(m, cmap='Greys')
        plt.scatter(x_positions, y_positions, c='r', marker='.')
        plt.scatter(xt_list[i][0], xt_list[i][1], c='g', marker='o')
        plt.savefig(f"images/step_{i+1}.png")
        list_of_plots.append(f"images/step_{i+1}.png")
        plt.close()
        Xt_1 = Xt

    return list_of_plots

def test_range_measurement_test():
    m = np.load('map/binary_image_cats.npy') 
    Xbar_t = [[400, 500, np.pi/2], [400, 450, np.pi/2], [400, 400, np.pi/2], [400, 350, np.pi/2], [400, 300, np.pi/2], [400, 250, np.pi/2]]
    zt = find_z_values_1(Xbar_t, m)
    x = 10
    print(mrl2.test_range_measurement(zt, Xbar_t, m , x))

if __name__ == "__main__":
    """
    See image: images/grid_localization.png for results
    """
    Grid_localization_test()

    """
    See GIF: images/animation_MCL_.gif for results
    """
    #MCL_test()
    create_GIF(MCL_test())

    """
    See GIF: images/animation_Augmented_MCL.gif for results
    """
    #Augmented_MCL_test()
    create_GIF(Augmented_MCL_test())

    """
    See GIF: images/animation_KLD_Sampling_MCL.gif for results
    """
    #KLD_Sampling_MCL_test()
    create_GIF(KLD_Sampling_MCL_test())

    """
    Prints out results of test_range_measurement
    """
    test_range_measurement_test()