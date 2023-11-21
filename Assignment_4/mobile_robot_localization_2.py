#!/usr/bin/env python3 

import numpy as np
import Motion_models as mm
import sensor_model as sm



def Grid_localization(pkt_1, ut, zt, m, delta_t, alpha, Theta, z_maxlen):
    """
    pkt_1 = how you want the grid to look like at time t-1
    ut = control input
    zt = sensor measurement
    m = map
    xt = robot pose
    xt_1 = robot pose at time t-1
    delta_t = time step
    alpha = motion error parameters
    Theta = sensor model parameters
    z_maxlen = maximum sensor range
    """
    
    n = 1
    size_y = m.shape[0]/pkt_1.shape[0]
    size_x = m.shape[1]/pkt_1.shape[1]
    xk = [size_x, size_y, np.pi/2] 
    xi = [size_x-ut[0]-1, size_y-1, np.pi/2] 
    p_kt = np.zeros(pkt_1.shape)
    for x in range(pkt_1.shape[0]-1):
        for y in range(pkt_1.shape[1]-1):
            p_bar_kt = pkt_1[x][y] * mm.motion_model_velocity(xk, ut, xi, delta_t, alpha)
            print('mm: ',mm.motion_model_velocity(xk, ut, xi, delta_t, alpha))
            p_kt[x][y] = n * p_bar_kt * sm.beam_range_finder_model(zt, xk, m, Theta, z_maxlen, zt_start=0, zt_end=2*np.pi, transpose=True)
            xk[1] += size_y
            xi[1] += size_y
        xi[0] += size_x
        xk[0] += size_x
        xk[1] = 0
        xi[1] = 0

    return p_kt


def MCL(Xt_1, ut, zt, m):
    pass


def Augmented_MCL(Xt_1, ut, zt, m):
    pass


def KLD_Sampling_MCL(Xt_1, ut, zt, m, epsilon, delta):
    pass


def test_range_measurement(ztk, Xbart, m):
    z_short, z_max, z_rand, z_hit = 0.1, 0.1, 0.1, 0.7 #########
    sigma_hit = 0.1 #########
    lambda_short = 0.1 #########
    ztk_star = 0 #########
    zmax = 100 #########
    x = 0.5 #########
    M = 100 #########

    p, q = 0
    for m in range(M):
        p  += z_short * sm.p_short(ztk, ztk_star, lambda_short)
        q  += z_hit * sm.p_hit(ztk, ztk_star, zmax, sigma_hit) + z_short * sm.p_short(ztk, ztk_star, lambda_short) + z_max * sm.p_max(ztk, zmax) + z_rand * sm.p_rand(ztk, zmax)

    if p/q <= x:
        return True
    else:
        return False