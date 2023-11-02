#!/usr/bin/env python3 

import numpy as np
import matplotlib.pyplot as plt
import mobile_robot_localization_1 as mrl1


def EKF_localization_known_correspondences_test():
    time = 4
    μt_1 = [8, 8, 0]
    μt_1_1 = μt_1.copy()
    Σt_1 = np.array([[0.1, 0, 0],
                    [0, 0.1, 0],
                    [0, 0, 0.1]])
    alpha = [0.1, 0.01, 0.01, 0.1]
    sigma = [0.1, 0.1, 0.1]

    m = np.load('map/map_1.npy')
    landmarks = [[0, 16, 0], [0, 12, 1], [0, 8, 2], [0, 4, 3], [0, 0, 4], 
                [4, 0, 5], [8, 0, 6], [12, 0, 7], [16, 0, 8], 
                [16, 4, 9], [16, 8, 10], [16, 12, 11], [16, 16, 12],
                [12, 16, 13], [8, 16, 14], [4, 16, 15]] # [x, y, signature]
    plt.imshow(m, cmap='gray')
    for i in landmarks:
        plt.plot(i[0], i[1], 'ro')
    
    """
    Need to define better u, z, and c
    """
    u = np.array([[1, 0], 
                  [1, np.pi/2],
                [1, np.pi/2],
                [1, np.pi/2]]) # [vt, wt]
    
    z = np.array([[[4, np.pi/2, 0], [8, np.pi/2, 1], [12, np.pi/2, 2], [16, np.pi/2, 3]],
                [[4, np.pi/2, 0], [8, np.pi/2, 1], [12, np.pi/2, 2], [16, np.pi/2, 3]],
                [[4, np.pi/2, 0], [8, np.pi/2, 1], [12, np.pi/2, 2], [16, np.pi/2, 3]],
                [[4, np.pi/2, 0], [8, np.pi/2, 1], [12, np.pi/2, 2], [16, np.pi/2, 3]]]) # [[rti, phiti, sti], ...]

    c = np.array([[[0, 16, 0], [0, 12, 1], [0, 8, 2], [0, 4, 3]], 
                  [[0, 0, 4], [4, 0, 5], [8, 0, 6], [12, 0, 7]],
                [[16, 0, 8], [16, 4, 9], [16, 8, 10], [16, 12, 11]],
                [[16, 16, 12], [12, 16, 13], [8, 16, 14], [4, 16, 15]]]) # [[x, y, signature], ...]

    for t in range(time):
        delta_t = 1 # Time difference between t-1 and t
        ut = u[t]
        zt = z[t]
        ct = c[t]
        μt_1, Σt_1, pzt = mrl1.EKF_localization_known_correspondences(μt_1, Σt_1, ut, zt, ct, delta_t, alpha, sigma)
        
        # For plotting estimate trajectory
        plt.plot(μt_1[0], μt_1[1], 'go')

        # For plotting true trajectory
        μt_1_1[0] = μt_1_1[0] + ut[0] * delta_t * np.cos(μt_1_1[2])
        μt_1_1[1] = μt_1_1[1] + ut[0] * delta_t * np.sin(μt_1_1[2])
        μt_1_1[2] = μt_1_1[2] + ut[1] * delta_t
        plt.plot(μt_1_1[0], μt_1_1[1], 'bo')

        print("Σt_1 = ", Σt_1)
        print("pzt = ", 0)
    
    plt.show()






if __name__ == '__main__':
    EKF_localization_known_correspondences_test()
