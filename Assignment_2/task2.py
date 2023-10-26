#!/usr/bin/env python3 

import sensor_model
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def test_ray_casting_plot(xt=np.array([200, 480, np.pi/2]), m=np.load('map/binary_image_cats.npy'), number_of_readings=8, z_maxlen=500, zt_start=np.pi/2, zt_end=-np.pi/2, filename="test"):
    transpose = True
    plt.imshow(m, cmap='Greys')
    plt.plot(xt[0], xt[1], 'ro')
    x_list = []
    y_list = []
    x_list_st = []
    y_list_st = []
    tolerance = 0.1

    x, y, theta = xt # x, y, θ (robot pose)
    angle_inc = np.linspace(zt_start, zt_end, number_of_readings) # For lidar
    if transpose: # Transpose the map when x is y and y is x
        m = np.transpose(m)

    for angle in angle_inc:
        for i in range(int(z_maxlen)): # For every block in the map up to the max range
            m_x = x + i * np.cos(angle + theta) 
            m_y = y + i * -np.sin(angle + theta) # -sin is only nessasary when the map is wrong // else sin
            if tolerance < angle < tolerance:
                x_list_st.append(m_x)
                y_list_st.append(m_y)
            else:
                x_list.append(m_x)
                y_list.append(m_y)
            if m[round(m_x)][round(m_y)] == 1: # If the block is occupied
                break # Return the distance to the block
                
    plt.plot(x_list, y_list, 'b-')
    plt.plot(x_list_st, y_list_st, 'g-')
    plt.savefig(f"{filename}.png")
    plt.close()
    return f"{filename}.png"


def test_ray_casting_gif():
    number_of_readings = 10
    zt_start = np.pi/4
    zt_end = -np.pi/4
    z_maxlen = 500
    loaded_map = np.load('map/binary_image_cats.npy')
    filenames = []

    xt = np.array([30 * 7.2, 80 * 6, np.pi/2])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_1"))
    xt = np.array([30 * 7.2, 75 * 6, np.pi/2])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_2"))
    xt = np.array([30 * 7.2, 70 * 6, np.pi/2])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_3"))
    xt = np.array([30 * 7.2, 65 * 6, np.pi/2])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_4"))
    xt = np.array([30 * 7.2, 60 * 6, np.pi/2])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_5"))
    xt = np.array([30 * 7.2, 55 * 6, np.pi/2])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_6"))
    xt = np.array([30 * 7.2, 50 * 6, np.pi/2])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_7"))
    xt = np.array([33 * 7.2, 46 * 6, np.pi/3])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_8"))
    xt = np.array([36 * 7.2, 43 * 6, np.pi/6])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_9"))
    xt = np.array([40 * 7.2, 40 * 6, 0])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_10"))
    xt = np.array([41 * 7.2, 42 * 6, -np.pi/8])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_11"))
    xt = np.array([42 * 7.2, 44 * 6, -np.pi/4])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_12"))
    xt = np.array([43 * 7.2, 47 * 6, -np.pi/4])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_13"))
    xt = np.array([44 * 7.2, 49 * 6, -np.pi/8])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_14"))
    xt = np.array([45 * 7.2, 52 * 6, 0])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_15"))
    xt = np.array([53 * 7.2, 54 * 6, np.pi/12])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_16"))
    xt = np.array([60 * 7.2, 56 * 6, np.pi/6])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_17"))
    xt = np.array([68 * 7.2, 50 * 6, np.pi/4])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_18"))
    xt = np.array([75 * 7.2, 43 * 6, np.pi/4])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_19"))
    xt = np.array([83 * 7.2, 36 * 6, np.pi/4])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_20"))
    xt = np.array([90 * 7.2, 30 * 6, np.pi/4])
    filenames.append(test_ray_casting_plot(xt, loaded_map, number_of_readings, z_maxlen, zt_start, zt_end, filename="plot_21"))

    # Convert the individual plots to a GIF
    with Image.open(filenames[0]) as im:
        im.save("images/task2.gif", save_all=True, append_images=[Image.open(f) for f in filenames[1:]], duration=100, loop=0)

    # Remove the individual plot files
    for f in filenames:
        os.remove(f)


def test_beam_range_finder_model():
    xt=np.array([30 * 7.2, 80 * 6, np.pi/2])
    m = np.load('map/binary_image.npy')
    zt = np.array([182.0, 153.0, 500, 279.0, 331.0, 480.0, 153.0, 184])
    z_hit = 0.40
    z_short = 0.20
    z_max = 0.25
    z_rand =  0.15
    sigma_hit = 0.1
    lambda_short = 0.5

    theta = np.array([z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short])
    z_maxlen = 500
    zt_start = np.pi/4
    zt_end = -np.pi/4
    print(f'{100 * sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end)} %')

if __name__ == "__main__":
    # test_ray_casting_gif()
    test_ray_casting_gif()