#!/usr/bin/env python3 
import numpy as np
import EKF_slam as EKF_slam
import sensor_model as sensor_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def find_z_values(xt, m):
    zt = 0
    return zt

def calculate_rti_phiti_sti(j_x, j_y, x, y, theta):
    r_hat = np.sqrt((j_x - x)**2 + (j_y - y)**2)
    theta_hat = np.arctan2(j_y - y, j_x - x) - theta

    return r_hat, theta_hat

def plot_beams(xt, m, number_of_readings, z_maxlen, zt_start=np.pi/4, zt_end=-np.pi/4, transpose = True):
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

            if np.sqrt((m_x - x)**2 + (m_y - y)**2) > z_maxlen: # If the distance to the block is larger than the max range
                break

            if -tolerance < angle < tolerance:
                x_list_st.append(m_x)
                y_list_st.append(m_y)
            else:
                x_list.append(m_x)
                y_list.append(m_y)
            if m_x < 0 or m_y < 0 or m_x > m.shape[0]-1 or m_y > m.shape[1]-1: # If the block is outside the map
                break

            if m[round(m_x)][round(m_y)] == 1: # If the block is occupied
                break # Return the distance to the block

    return x_list, y_list, x_list_st, y_list_st 

def EKF_SLAM_known_correspondences_test():
    m = np.load('map/map_landmarks.npy')
    μt_1 = np.array([25, 20, np.pi/2])
    μt_1_1 = μt_1.copy()
    Σt_1 = np.eye(3) / 10
    ut = np.array([[4, 1], [4, 1], [4, 0.1], [4, 0.1]])
    zt = 0 ############################ [[[]],[]] [range, bearing, signature]
    ct = 0 ############################ Know from map witch id
    sigma = np.array([2, 0.1, 2])

    fig, ax = plt.subplots()
    ax.imshow(m, cmap='gray')
    plot_estimated= []
    plot_true = []

    for i, utt in enumerate(ut):
        #μt_1, Σt_1 = EKF_slam.EKF_SLAM_known_correspondences(μt_1, Σt_1, utt, zt, ct, len(ct), sigma)
        
        # For plotting the mean value
        plot_estimated.append([μt_1[0], μt_1[1]])
                
        # For plotting the uncertainty ellipse
        circle = patches.Ellipse((μt_1[0], μt_1[1]), np.sqrt(Σt_1[0][0]), np.sqrt(Σt_1[1][1]), np.sqrt(Σt_1[2][2]), edgecolor='green') # Std from Σt_1 is the sqrt of the diagonal elements
        ax.add_patch(circle)

        # For plotting true trajectory
        plot_true.append([μt_1_1[0], μt_1_1[1]])

        # PLotting beams, to know which landmarks the robot sees
        x_list, y_list, x_list_st, y_list_st = plot_beams(μt_1_1, m, 2, 100)
        ax.plot(x_list, y_list, 'r.')
        ax.plot(x_list_st, y_list_st, 'r.')

        μt_1_1[0] = μt_1_1[0] + utt[0] * 1 * np.cos(μt_1_1[2])
        μt_1_1[1] = μt_1_1[1] + utt[0] * 1 * -np.sin(μt_1_1[2])
        μt_1_1[2] = μt_1_1[2] + utt[1] * 1 

    x_estimated, y_estimated = zip(*plot_estimated)
    x_true, y_true = zip(*plot_true)
    ax.plot(x_true, y_true, 'bo', label='True trajectory')
    ax.plot(x_estimated, y_estimated, 'go', label='Estimated trajectory')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig('images/EKF_SLAM_known_correspondences.png')


if __name__ == '__main__':
    EKF_SLAM_known_correspondences_test()