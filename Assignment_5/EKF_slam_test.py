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


def plot_beams(xt, m, number_of_readings, z_maxlen, zt_start=np.pi/8, zt_end=-np.pi/8, transpose = True):
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
    N = 31
    μt_1 = np.array([25, 20, np.pi/2, 0, 0, 1, 5, 0, 2, 28, 0, 3, 36, 0, 4, 41, 0, 5, 83, 0, 6, 0, 4, 7, 83, 4, 8, 10, 8, 9, 15, 8, 10, 20, 8, 11, 25, 8, 12, 30, 8, 13, 35, 8, 14, 42, 8, 15, 48, 8, 16, 56, 8, 17, 0, 16, 18, 42, 18, 19, 6, 20, 20, 36, 22, 21, 48, 22, 22, 0, 24, 23, 12, 24, 24, 42, 26, 25, 83, 27, 26, 6, 28, 27, 83, 29, 28, 10, 31, 29, 24, 31, 30, 83, 31, 31])
    μt_1_1 = μt_1.copy()
    μt_1 = μt_1.reshape((3+3*N, 1))
    μ_bar_j = np.array([0]*(3+3*N))
    Σt_1 = np.zeros((3+3*N, 3+3*N))
    Σt_1[0, 0] = 1/3
    Σt_1[1, 1] = 1/3
    Σt_1[2, 2] = 1/3
    ut = np.array([[4, 1], [4, 1], [4, 0.1], [4, 0.1]])
    
    zt = np.array([[[13.0, -3.5, 11], [12.0, -3.14, 12], [20.22, -3, 3],[13.0, -2.746, 13]],
          [[27.73, -5.26, 7], [29.68, -5.143, 1], [25.61, -5.03, 2], [17.0, -5.22, 9], [12.80, -5.037, 10]],
          [[21.74, -0.528, 18], [16.8, -0.8, 20], [23.90, -0.869, 23], [14, -1.24, 24], [21, -1.16, 27]],
          [[12.81, -0.88, 20], [19.90, -0.97, 23], [10.39, -1.48, 24], [17.32, -1.34, 27]]], dtype=object)
    
    ct = np.array([[[20, 8, 11], [25, 8, 12], [28, 0, 3], [30, 8, 13]], 
                [[0, 4, 7], [0, 0, 1], [5, 0, 2], [10, 8, 9] , [15, 8, 10]],
                [[0, 16, 18],[6, 20, 20],[0, 24, 23], [12, 24, 24],[6, 28, 27]],
                [[6, 20, 20],[0, 24, 23], [12, 24, 24],[6, 28, 27]]], dtype=object)
    
    sigma = np.array([2, 0.1, 2])

    fig, ax = plt.subplots()
    ax.imshow(m, cmap='gray')
    plot_estimated= []
    plot_true = []

    for i, utt in enumerate(ut):
        μt_1, Σt_1, μ_bar_j = EKF_slam.EKF_SLAM_known_correspondences(μt_1, Σt_1, utt, zt[i], ct[i], N, μ_bar_j, sigma)
        
        # For plotting the mean value
        plot_estimated.append([μt_1[0], μt_1[1]])
                
        # For plotting the uncertainty ellipse
        circle = patches.Ellipse((μt_1[0], μt_1[1]), np.sqrt(Σt_1[0][0]), np.sqrt(Σt_1[1][1]), np.sqrt(Σt_1[2][2]), edgecolor='green') # Std from Σt_1 is the sqrt of the diagonal elements
        ax.add_patch(circle)

        # For plotting true trajectory
        plot_true.append([μt_1_1[0], μt_1_1[1]])
        μt_1_1[0] = μt_1_1[0] + utt[0] * 1 * np.cos(μt_1_1[2])
        μt_1_1[1] = μt_1_1[1] + utt[0] * 1 * -np.sin(μt_1_1[2])
        μt_1_1[2] = μt_1_1[2] + utt[1] * 1 

    x_estimated, y_estimated = zip(*plot_estimated)
    x_true, y_true = zip(*plot_true)
    ax.plot(x_true, y_true, 'bo', label='True trajectory')
    ax.plot(x_estimated, y_estimated, 'go', label='Estimated trajectory')
    ax.legend(bbox_to_anchor=(0.80, 1.30), loc='upper center')
    plt.savefig('images/EKF_SLAM_known_correspondences.png')


def EKF_SLAM_test():
    m = np.load('map/map_landmarks.npy')
    N = 31
    μt_1 = np.array([25, 20, np.pi/2, 0, 0, 1, 5, 0, 2, 28, 0, 3, 36, 0, 4, 41, 0, 5, 83, 0, 6, 0, 4, 7, 83, 4, 8, 10, 8, 9, 15, 8, 10, 20, 8, 11, 25, 8, 12, 30, 8, 13, 35, 8, 14, 42, 8, 15, 48, 8, 16, 56, 8, 17, 0, 16, 18, 42, 18, 19, 6, 20, 20, 36, 22, 21, 48, 22, 22, 0, 24, 23, 12, 24, 24, 42, 26, 25, 83, 27, 26, 6, 28, 27, 83, 29, 28, 10, 31, 29, 24, 31, 30, 83, 31, 31])
    μt_1_1 = μt_1.copy()
    μt_1 = μt_1.reshape((3+3*N, 1))
    μ_bar_Nt1 = np.array([0]*(3+3*N))
    Σt_1 = np.zeros((3+3*N, 3+3*N))
    Σt_1[0, 0] = 1/3
    Σt_1[1, 1] = 1/3
    Σt_1[2, 2] = 1/3
    ut = np.array([[4, 1], [4, 1], [4, 0.1], [4, 0.1]])
    
    zt = np.array([[[13.0, -3.5, 11], [12.0, -3.14, 12], [20.22, -3, 3],[13.0, -2.746, 13]],
          [[27.73, -5.26, 7], [29.68, -5.143, 1], [25.61, -5.03, 2], [17.0, -5.22, 9], [12.80, -5.037, 10]],
          [[21.74, -0.528, 18], [16.8, -0.8, 20], [23.90, -0.869, 23], [14, -1.24, 24], [21, -1.16, 27]],
          [[12.81, -0.88, 20], [19.90, -0.97, 23], [10.39, -1.48, 24], [17.32, -1.34, 27]]], dtype=object)
    
    ct = np.array([[[20, 8, 11], [25, 8, 12], [28, 0, 3], [30, 8, 13]], 
                [[0, 4, 7], [0, 0, 1], [5, 0, 2], [10, 8, 9] , [15, 8, 10]],
                [[0, 16, 18],[6, 20, 20],[0, 24, 23], [12, 24, 24],[6, 28, 27]],
                [[6, 20, 20],[0, 24, 23], [12, 24, 24],[6, 28, 27]]], dtype=object)
    
    sigma = np.array([2, 0.1, 2])

    fig, ax = plt.subplots()
    ax.imshow(m, cmap='gray')
    plot_estimated= []
    plot_true = []

    for i, utt in enumerate(ut):
        μt_1, Σt_1, μ_bar_Nt1 = EKF_slam.EKF_SLAM(μt_1, Σt_1, utt, zt[i], N, μ_bar_Nt1, sigma)
        
        # For plotting the mean value
        plot_estimated.append([μt_1[0], μt_1[1]])
                
        # For plotting the uncertainty ellipse
        circle = patches.Ellipse((μt_1[0], μt_1[1]), np.sqrt(Σt_1[0][0]), np.sqrt(Σt_1[1][1]), np.sqrt(Σt_1[2][2]), edgecolor='green') # Std from Σt_1 is the sqrt of the diagonal elements
        ax.add_patch(circle)

        # For plotting true trajectory
        plot_true.append([μt_1_1[0], μt_1_1[1]])
        μt_1_1[0] = μt_1_1[0] + utt[0] * 1 * np.cos(μt_1_1[2])
        μt_1_1[1] = μt_1_1[1] + utt[0] * 1 * -np.sin(μt_1_1[2])
        μt_1_1[2] = μt_1_1[2] + utt[1] * 1 

    x_estimated, y_estimated = zip(*plot_estimated)
    x_true, y_true = zip(*plot_true)
    ax.plot(x_true, y_true, 'b.', label='True trajectory')
    ax.plot(x_estimated, y_estimated, 'g.', label='Estimated trajectory')
    ax.legend(bbox_to_anchor=(0.80, 1.30), loc='upper center')
    plt.savefig('images/EKF_SLAM.png')


if __name__ == '__main__':
    """
    See image EKF_SLAM_known_correspondences.png for the result of the EKF_SLAM_known_correspondences_test()
    """
    EKF_SLAM_known_correspondences_test()

    """
    See image EKF_SLAM.png for the result of the EKF_SLAM_test()
    """
    EKF_SLAM_test()
