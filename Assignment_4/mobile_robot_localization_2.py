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
            p_bar_kt =  pkt_1[x][y] * mm.motion_model_velocity(xk, ut, xi, delta_t, alpha)
            p_kt[x][y] = n * p_bar_kt * sm.beam_range_finder_model(zt, xk, m, Theta, z_maxlen, zt_start=0, zt_end=2*np.pi, transpose=True)
            xk[1] += size_y
            xi[1] += size_y
        xi[0] += size_x
        xk[0] += size_x
        xk[1] = 0
        xi[1] = 0
    return p_kt

def MCL(Xt_1, ut, zt, m):
    alpha = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    theta = np.array([0.8157123986145881, 0.00235666025958796, 0.16552184295092348, 0.01640909817490046, 1.9665518618953464, 0.0029480342354130016])
    z_maxlen = 500
    sense_coord = np.array([0, 0, 0])
    zt_start = np.pi/4
    zt_end = -np.pi/4
    Xbar_t = []

    for m_, xt_1 in enumerate(Xt_1): # is number of particles
        
        xt = mm.sample_model_velocity(ut, xt_1[0], 1, alpha)
        while xt[0] < 0 or xt[0] > m.shape[1] or xt[1] < 0 or xt[1] > m.shape[0] or m[int(xt[1]), int(xt[0])] == 1:
            xt = mm.sample_model_velocity(ut, xt_1[0], 1, alpha)
        wt = sm.likelihood_field_range_finder_model(zt, xt, m, theta, z_maxlen, sense_coord, zt_start, zt_end, transpose=False) 
        Xbar_t.append([xt, wt])

        # if weight is high, sample more around that particle
        if xt_1[1] > 0.1:
            for _ in range(5):
                xt = mm.sample_model_velocity(ut, xt_1[0], 1, alpha)
                while xt[0] < 0 or xt[0] > m.shape[1] or xt[1] < 0 or xt[1] > m.shape[0] or m[int(xt[1]), int(xt[0])] == 1:
                    xt = mm.sample_model_velocity(ut, xt_1[0], 1, alpha)
                wt = sm.likelihood_field_range_finder_model(zt, xt, m, theta, z_maxlen, sense_coord, zt_start, zt_end, transpose=False) 
                Xbar_t.append([xt, wt])

        while len(Xbar_t) > len(Xt_1): # if too many particles, remove the ones with the lowest weight
            min_weight_sublist = min(Xbar_t, key=lambda x: x[1])
            Xbar_t.remove(min_weight_sublist)

    Xbar_t = np.array(Xbar_t, dtype=object)
    epsilon = 1e-10  # Small constant to avoid division by zero
    Xbar_t[:, 1] = Xbar_t[:, 1] / (np.sum(Xbar_t[:, 1]) + epsilon)

    return np.array(Xbar_t)


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