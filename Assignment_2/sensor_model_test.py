#!/usr/bin/env/python3

import sensor_model
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def test_p_hit():
    zmax = 30
    z = np.linspace(0, zmax, 1000)
    ztk_star = 20
    sigma_hit = 4
    list_av_values = []
    for ztk in z:
        list_av_values.append(sensor_model.p_hit(ztk, ztk_star, zmax, sigma_hit))
    plt.title("p_hit")
    plt.plot(z, list_av_values)
    plt.show()


def test_p_short():
    zmax = 30
    z = np.linspace(0, zmax, 1000)
    ztk_star = 20
    lambda_short = 0.1
    list_av_values = []
    for ztk in z:
        list_av_values.append(sensor_model.p_short(ztk, ztk_star, lambda_short))
    plt.title("p_short")
    plt.plot(z, list_av_values)
    plt.show()


def test_p_max():
    zmax = 30
    z = np.linspace(0, zmax, 1000)
    list_av_values = []
    for ztk in z:
        list_av_values.append(sensor_model.p_max(ztk, zmax))
    plt.title("p_max")
    plt.plot(z, list_av_values)
    plt.show()


def test_p_rand():
    zmax = 30
    z = np.linspace(0, zmax, 1000)
    list_av_values = []
    for ztk in z:
        list_av_values.append(sensor_model.p_rand(ztk, zmax))
    plt.title("p_rand")
    plt.plot(z, list_av_values)
    plt.show()


def test_all_p():
    zmax = 30
    z = np.linspace(0, zmax, 1000)
    ztk_star = 20
    list_av_values = []
    z_hit = 0.50
    z_short = 0.25
    z_max = 0.05
    z_rand =  0.2
    sigma_hit = 2
    lambda_short = 0.1

    for ztk in z:
        value = 0
        value += z_hit * (sensor_model.p_hit(ztk, ztk_star, zmax, sigma_hit))
        value += z_short * (sensor_model.p_short(ztk, ztk_star, lambda_short))
        value += z_max * (sensor_model.p_max(ztk, zmax))
        value += z_rand * (sensor_model.p_rand(ztk, zmax))
        list_av_values.append(value)
    plt.title("p_all")
    plt.plot(z, list_av_values)
    plt.show()  


def test_ray_casting():
    xt = np.array([30 * 7.2, 80 * 6, np.pi/2])
    loaded_map = np.load('map/binary_image.npy')
    z_maxlen = 500
    zt_start = np.pi/4
    zt_end = -np.pi/4
    number_of_readings = 8
    print(f"Ray casting values :{[sensor_model.ray_casting(xt, loaded_map, theta, z_maxlen, transpose=True) for theta in np.linspace(zt_start, zt_end, number_of_readings)]}")


def test_ray_casting_plot(xt=np.array([200, 480, np.pi/2]), m=np.load('map/binary_image.npy'), number_of_readings=8, z_maxlen=500, zt_start=np.pi/2, zt_end=-np.pi/2, filename="test"):
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
    plt.savefig(f"images/{filename}.png")
    plt.close()
    return f"images/{filename}.png"


def test_ray_casting_gif():
    number_of_readings = 10
    zt_start = np.pi/4
    zt_end = -np.pi/4
    z_maxlen = 500
    loaded_map = np.load('map/binary_image.npy')
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
        im.save("images/animation_front_laser.gif", save_all=True, append_images=[Image.open(f) for f in filenames[1:]], duration=100, loop=0)

    # Remove the individual plot files
    for f in filenames:
        os.remove(f)


def test_beam_range_finder_model():
    m = np.load('map/binary_image.npy')
    m = np.transpose(m) 
    zt = np.array([182.0, 153.0, 500, 279.0, 331.0, 480.0, 153, 184]) # exact values 
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
    zt_start = np.pi/4
    zt_end = -np.pi/4
    xt=np.array([30 * 7.2, 80 * 6, np.pi/2])
    print(f"Beam range finder model :{sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end)}")

def test_learn_intrinsic_parameters():
    Z = np.array([329, 85.0, 500, 182.0])
    noise = np.random.normal(0, 2, size=len(Z))
    Z += noise
    Z[2] += 500 # to not get larger than 500 

    X=np.array([[30 * 7.2, 80 * 6, np.pi/2],[400, 200, 0],
                [600, 300, np.pi],[30 * 7.2, 80 * 6, 3*np.pi/4]])
    m = np.load('map/binary_image.npy')
    z_maxlen = 500
    angle = 0
    Theta = [0.5, 0.1, 0.2, 0.2, 5, 0.5]
    values = sensor_model.learn_intrinsic_parameters(Z, X, m, Theta, z_maxlen, angle)
    print(f"All params :{values}, Sum of z values :{sum(values[:4])}")


def test_likelihood_field_range_finder_model():
    zt = np.array([182.0, 153.0, 500, 279.0, 331.0, 480.0, 153, 184])
    noise = np.random.normal(0, 2, size=len(zt))
    zt += noise
    zt[2] = 500 # to not get larger than 500

    xt=np.array([30 * 7.2, 80 * 6, np.pi/2])
    m = np.load('map/binary_image_cats.npy')
    theta = np.array([0.9656165275012334, 0.013197869913444309, 0, 0.021185602585322268, 2.277803215809316, 0.003070493203341887])
    z_maxlen = 500
    sense_coord = np.array([0, 0, 0])
    print(f"Likelihood field :{100 * sensor_model.likelihood_field_range_finder_model(zt, xt, m, theta, z_maxlen, sense_coord, transpose=True)} %")

def test_landmark_model_known_correspondence():
    fit = [350, -2.5, 1] #[rit, thetait, sit]
    cit = [400, 200, 1] #[jx, jy, ji] 
    xt=np.array([30 * 7.2, 80 * 6, np.pi/2])
    sigma = [0.1, 0.05, 0.1] # sigma_r = 0.1, sigma_theta = 0.05, sigma_s = 0.1
    print(f'Land mark model :{100 * sensor_model.landmark_model_known_correspondence(fit, cit, xt, sigma)} %')


def test_sample_landmark_model_known_correspondence():
    fit = [355, -2.5, 1] #[rit, thetait, sit]
    cit = [400, 200, 1] #[jx, jy, ji]
    m = np.load('map/binary_image.npy')
    sigma = [0.1, 0.05] # sigma_r = 0.1, sigma_theta = 0.05, sigma_s = 0.1

    x_positions, y_positions, z_positions = zip(*(sensor_model.sample_landmark_model_known_correspondence(fit, cit, m, sigma) for _ in range(10000)))

    plt.title("Sample landmark model")
    plt.imshow(np.load('map/binary_image.npy'), cmap='Greys')
    plt.scatter(x_positions, y_positions, c='b', marker='o')
    plt.scatter(400,200, c='r', marker='o')
    plt.scatter(216,480, c='r', marker='o')
    plt.show()
    # Should be [216.0, 480, 1.57]


if __name__ == "__main__":

    #test_p_hit()
    #test_p_short()
    #test_p_max()
    #test_p_rand()
    #test_all_p()
    #test_ray_casting()
    #test_ray_casting_plot()
    #test_ray_casting_gif()
    #test_beam_range_finder_model()
    #test_learn_intrinsic_parameters()
    #test_likelihood_field_range_finder_model()
    #test_landmark_model_known_correspondence()
    ##test_sample_landmark_model_known_correspondence()
    pass
