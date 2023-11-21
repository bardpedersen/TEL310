#!/usr/bin/env python3 

import numpy as np
import mobile_robot_localization_2 as mrl2

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
    print(p_kt)
    print(np.sum(p_kt))
    pass 


if __name__ == "__main__":
    Grid_localization_test()