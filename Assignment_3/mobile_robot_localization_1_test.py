#!/usr/bin/env python3 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mobile_robot_localization_1 as mrl1


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
            if m_x < 0 or m_y < 0 or m_x > 16 or m_y > 16: # If the block is outside the map
                break

            if m[round(m_x)][round(m_y)] == 1: # If the block is occupied
                break # Return the distance to the block

    return x_list, y_list, x_list_st, y_list_st 


def EKF_localization_known_correspondences_test():
    time = 4
    μt_1 = [8, 8, 0]
    μt_1_1 = μt_1.copy() # Create a copy of μt_1 for plotting true trajectory
    
    Σt_1 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    
    alpha = [1, 1, 1, 1]
    sigma = [1, 1, 1]

    m = np.load('map/map_1.npy')
    landmarks = [[0, 16, 0], [0, 12, 1], [0, 8, 2], [0, 4, 3], [0, 0, 4], 
                [4, 0, 5], [8, 0, 6], [12, 0, 7], [16, 0, 8], 
                [16, 4, 9], [16, 8, 10], [16, 12, 11], [16, 16, 12],
                [12, 16, 13], [8, 16, 14], [4, 16, 15]] # [x, y, signature]
    
    fig, ax = plt.subplots()
    ax.imshow(m, cmap='gray')
    plot_landmarks = []
    plot_estimated= []
    plot_true = []
    for i in landmarks:
        plot_landmarks.append([i[0], i[1]])

    u = np.array([[3, np.pi/4], 
                [4, np.pi/4],
                [4, np.pi/3],
                [2, np.pi/2]]) # [vt, wt]
    
    c = np.array([[[16, 16, 12], [16, 12, 11], [16, 8, 10], [16, 4, 9], [16, 0, 8]],
                [[16, 8, 10], [16, 4, 9], [16, 0, 8], [12, 0, 7]],
                [[16, 0, 8], [12, 0, 7]],
                [[12, 0, 7], [8, 0, 6], [4, 0, 5], [0, 0, 4], [0, 4, 3]]], dtype=object) # [[x, y, signature], ...]
    
    z = np.array([[[11, 0.78, 12], [9, 0.46, 11], [8, 0, 10], [9, -0.46, 9], [11, -0.78, 8]],
                [[5, -0.78, 10], [6.4, -1.5, 9], [9.4, -1.8, 8], [8, -2.2, 7]],
                [[5.6, -2.7, 8], [5.5, -3.5, 7]],
                [[2, -5, 7], [5.9, -5.5, 6], [9.9, -5.6, 5], [13.8, -5.6, 4], [14.1, 0.32, 3]]], dtype=object) # [[rti, phiti, sti], ...]

    for t in range(time):
        delta_t = 1 # Time difference between t-1 and t
        ut = u[t]
        zt = z[t]
        ct = c[t]

        μt_1, Σt_1, pzt = mrl1.EKF_localization_known_correspondences(μt_1, Σt_1, ut, zt, ct, delta_t, alpha, sigma)
        
        # For plotting the mean value
        plot_estimated.append([μt_1[0], μt_1[1]])
                
        # For plotting the uncertainty ellipse
        circle = patches.Ellipse((μt_1[0], μt_1[1]), np.sqrt(Σt_1[0][0]), np.sqrt(Σt_1[1][1]), np.sqrt(Σt_1[2][2]), edgecolor='green') # Std from Σt_1 is the sqrt of the diagonal elements
        ax.add_patch(circle)
        print("pzt = ", pzt)

        # For plotting true trajectory
        plot_true.append([μt_1_1[0], μt_1_1[1]])

        # For calculating rti, phiti
        #for j in ct:
        #    print(calculate_rti_phiti_sti(j[0], j[1], μt_1_1[0], μt_1_1[1], μt_1_1[2]))

        μt_1_1[0] = μt_1_1[0] + ut[0] * delta_t * np.cos(μt_1_1[2])
        μt_1_1[1] = μt_1_1[1] + ut[0] * delta_t * -np.sin(μt_1_1[2])
        μt_1_1[2] = μt_1_1[2] + ut[1] * delta_t
        
        # PLotting beams, to know which landmarks the robot sees
        #x_list, y_list, x_list_st, y_list_st = plot_beams(μt_1_1, m, 2, 16)
        #ax.plot(x_list, y_list, 'b.')
        #ax.plot(x_list_st, y_list_st, 'g.')

    x_landmarks, y_landmarks = zip(*plot_landmarks)
    x_estimated, y_estimated = zip(*plot_estimated)
    x_true, y_true = zip(*plot_true)
    ax.plot(x_landmarks, y_landmarks, 'ro', label='Landmarks')
    ax.plot(x_true, y_true, 'bo', label='True trajectory')
    ax.plot(x_estimated, y_estimated, 'go', label='Estimated trajectory')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid()
    plt.show()


def EKF_localization_test():
    time = 4
    μt_1 = [8, 8, 0]
    μt_1_1 = μt_1.copy() # Create a copy of μt_1 for plotting true trajectory
    
    Σt_1 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    
    alpha = [1, 1, 1, 1]
    sigma = [1, 1, 1]

    m = np.load('map/map_1.npy')
    landmarks = [[0, 16, 0], [0, 12, 1], [0, 8, 2], [0, 4, 3], [0, 0, 4], 
                [4, 0, 5], [8, 0, 6], [12, 0, 7], [16, 0, 8], 
                [16, 4, 9], [16, 8, 10], [16, 12, 11], [16, 16, 12],
                [12, 16, 13], [8, 16, 14], [4, 16, 15]] # [x, y, signature]
    
    
    fig, ax = plt.subplots()
    ax.imshow(m, cmap='gray')
    plot_landmarks = []
    plot_estimated= []
    plot_true = []
    for i in landmarks:
        plot_landmarks.append([i[0], i[1]])

    u = np.array([[3, np.pi/4], 
                [4, np.pi/4],
                [4, np.pi/3],
                [2, np.pi/2]]) # [vt, wt]
    
    z = np.array([[[11, 0.78, 12], [9, 0.46, 11], [8, 0, 10], [9, -0.46, 9], [11, -0.78, 8]],
                [[5, -0.78, 10], [6.4, -1.5, 9], [9.4, -1.8, 8], [8, -2.2, 7]],
                [[5.6, -2.7, 8], [5.5, -3.5, 7]],
                [[2, -5, 7], [5.9, -5.5, 6], [9.9, -5.6, 5], [13.8, -5.6, 4], [14.1, 0.32, 3]]], dtype=object) # [[rti, phiti, sti], ...]

    for t in range(time):
        delta_t = 1 # Time difference between t-1 and t
        ut = u[t]
        zt = z[t]

        μt_1, Σt_1 = mrl1.EKF_localization(μt_1, Σt_1, ut, zt, landmarks, delta_t, alpha, sigma)
        
        # For plotting the mean value
        plot_estimated.append([μt_1[0], μt_1[1]])
                
        # For plotting the uncertainty ellipse
        circle = patches.Ellipse((μt_1[0], μt_1[1]), np.sqrt(Σt_1[0][0]), np.sqrt(Σt_1[1][1]), np.sqrt(Σt_1[2][2]), edgecolor='green') # Std from Σt_1 is the sqrt of the diagonal elements
        ax.add_patch(circle)

        # For plotting true trajectory
        plot_true.append([μt_1_1[0], μt_1_1[1]])

        # For calculating rti, phiti
        #for j in ct:
        #    print(calculate_rti_phiti_sti(j[0], j[1], μt_1_1[0], μt_1_1[1], μt_1_1[2]))

        μt_1_1[0] = μt_1_1[0] + ut[0] * delta_t * np.cos(μt_1_1[2])
        μt_1_1[1] = μt_1_1[1] + ut[0] * delta_t * -np.sin(μt_1_1[2])
        μt_1_1[2] = μt_1_1[2] + ut[1] * delta_t
        
        # PLotting beams, to know which landmarks the robot sees
        #x_list, y_list, x_list_st, y_list_st = plot_beams(μt_1_1, m, 2, 16)
        #ax.plot(x_list, y_list, 'b.')
        #ax.plot(x_list_st, y_list_st, 'g.')

    x_landmarks, y_landmarks = zip(*plot_landmarks)
    x_estimated, y_estimated = zip(*plot_estimated)
    x_true, y_true = zip(*plot_true)
    ax.plot(x_landmarks, y_landmarks, 'ro', label='Landmarks')
    ax.plot(x_true, y_true, 'bo', label='True trajectory')
    ax.plot(x_estimated, y_estimated, 'go', label='Estimated trajectory')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid()
    plt.show()


def UKF_localization_test():
    μt_1 = [8, 8, 0]
    Σt_1 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    alpha = [1, 1, 1, 1]
    sigma = [1, 1, 1]

    u = np.array([[3, np.pi/4], 
                [4, np.pi/4],
                [4, np.pi/3],
                [2, np.pi/2]]) # [vt, wt]
    
    z = np.array([[[11, 0.78, 12], [9, 0.46, 11], [8, 0, 10], [9, -0.46, 9], [11, -0.78, 8]],
                [[5, -0.78, 10], [6.4, -1.5, 9], [9.4, -1.8, 8], [8, -2.2, 7]],
                [[5.6, -2.7, 8], [5.5, -3.5, 7]],
                [[2, -5, 7], [5.9, -5.5, 6], [9.9, -5.6, 5], [13.8, -5.6, 4], [14.1, 0.32, 3]]], dtype=object) # [[rti, phiti, sti], ...]

    for t in range(4):
        ut = u[t]
        zt = z[t]
        mrl1.UKF_localization(μt_1, Σt_1, ut, zt, alpha, sigma)

if __name__ == '__main__':
    #EKF_localization_known_correspondences_test()
    #EKF_localization_test()
    UKF_localization_test()
    