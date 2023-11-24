#!/usr/bin/env python3 

import numpy as np
import Motion_models as mm
import sensor_model as sm



def Grid_localization(pkt_1, ut, zt, m, delta_t, alpha, Theta, z_maxlen):
    """
    Function that implements the Grid Localization algorithm
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

    Returns a grid with the probability of the robot being at each cell
    """
    sense_coord = np.array([0, 0, 0])
    zt_start = 0
    zt_end = np.pi
    size_y = m.shape[0]/pkt_1.shape[0]
    size_x = m.shape[1]/pkt_1.shape[1]
    xk = [size_x/2, size_y/2, np.pi/2] 
    xi = [size_x/2 + ut[0], size_y/2 - 1, np.pi/2] 
    p_kt = np.zeros(pkt_1.shape)
    for x in range(pkt_1.shape[0]):
        for y in range(pkt_1.shape[1]):
            p_bar_kt =  pkt_1[x][y] * mm.motion_model_velocity(xk, ut, xi, delta_t, alpha) ########
            p_kt[x][y] = p_bar_kt * sm.likelihood_field_range_finder_model(zt, xk, m, Theta, z_maxlen, sense_coord, zt_start, zt_end, transpose=True) 
            xk[1] += size_y
            xi[1] += size_y
        xi[0] += size_x
        xk[0] += size_x
        xk[1] = 0
        xi[1] = 0

    #normalize p_kt
    p_kt = p_kt/np.sum(p_kt)
    return p_kt

def generate_particles(m):
    """
    Function that generates a random particle on the free spaces in the map m
    m = map

    Returns a random particle with the following format: [x, y, theta]
    """
    free_spaces = np.argwhere(m == 0)  # Get indices of free spaces
    random_index = free_spaces[np.random.choice(free_spaces.shape[0])]  # Randomly select one index
    random_theta = np.random.uniform(0, 2*np.pi)  # Generate a random theta

    return [random_index[1], random_index[0], random_theta]

def MCL(Xt_1, ut, zt, m):
    """
    Function that implements the Monte Carlo Localization algorithm
    Xt_1 = list of particles at time t-1
    ut = control input
    zt = sensor measurement
    m = map

    Returns a list of particles with the following format: [[x, y, theta], weight]
    """
    alpha = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    theta = np.array([0.8157123986145881, 0.00235666025958796, 0.16552184295092348, 0.01640909817490046, 1.9665518618953464, 0.0029480342354130016])
    z_maxlen = 500
    sense_coord = np.array([0, 0, 0])
    zt_start = 0
    zt_end = np.pi
    Xbar_t = []

    for m_, xt_1 in enumerate(Xt_1): # is number of particles
        xt = mm.sample_model_velocity(ut, xt_1[0], 1, alpha)
        wt = sm.likelihood_field_range_finder_model(zt, xt, m, theta, z_maxlen, sense_coord, zt_start, zt_end, transpose=True) 
        Xbar_t.append([xt, wt])
        
    # Normalize the weights
    sum_weights = sum(w for (_, w) in Xbar_t)
    normalized_weights = [w / sum_weights for (_, w) in Xbar_t]

    # Resampling
    indices = np.random.choice(range(len(Xt_1)), len(Xt_1), p=normalized_weights)
    Xbar_t = [[Xbar_t[i][0], Xbar_t[i][1]] for i in indices]
    return np.array(Xbar_t)


def Augmented_MCL(Xt_1, ut, zt, m):
    """
    Function that implements the Augmented Monte Carlo Localization algorithm
    Xt_1 = list of particles at time t-1
    ut = control input
    zt = sensor measurement
    m = map

    Returns a list of particles with the following format: [[x, y, theta], weight]
    """
    alpha = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    theta = np.array([0.8157123986145881, 0.00235666025958796, 0.16552184295092348, 0.01640909817490046, 1.9665518618953464, 0.0029480342354130016])
    z_maxlen = 500
    sense_coord = np.array([0, 0, 0])
    zt_start = 0
    zt_end = np.pi
    Xbar_t = []
    Xt = []
    Wavg = 0
    Wslow = 0.1
    Wfast = 0.6
    alpha_slow = 0.001
    alpha_fast = 0.1

    for m_, xt_1 in enumerate(Xt_1): # is number of particles
        xt = mm.sample_model_velocity(ut, xt_1[0], 1, alpha)
        wt = sm.likelihood_field_range_finder_model(zt, xt, m, theta, z_maxlen, sense_coord, zt_start, zt_end, transpose=True) 
        Xbar_t.append([xt, wt])
        Wavg += wt/len(Xt_1)

    # Normalize the weights
    Wslow  += alpha_slow * (Wavg - Wslow)
    Wfast  += alpha_fast * (Wavg - Wfast)

    for m_, (xt, wt) in enumerate(Xbar_t):
        if np.random.rand() < max(0.0, 1.0 - Wfast / Wslow):
            Xt.append([generate_particles(m), 1/len(Xt_1)])
        else:
            indices = np.random.choice(range(len(Xbar_t)), len(Xbar_t), p=[Wavg / sum(w for (_, w) in Xbar_t) for _ in range(len(Xbar_t))])
            Xt.append([Xbar_t[indices[0]][0], Xbar_t[indices[0]][1]])

    return np.array(Xt)


def KLD_Sampling_MCL(Xt_1, ut, zt, m, epsilon, delta):
    """
    Function that implements the KLD Sampling Monte Carlo Localization algorithm
    Xt_1 = list of particles at time t-1
    ut = control input
    zt = sensor measurement
    m = map
    epsilon = threshold
    delta = threshold

    Returns a list of particles with the following format: [[x, y, theta], weight]
    """
    alpha = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    theta = np.array([0.8157123986145881, 0.00235666025958796, 0.16552184295092348, 0.01640909817490046, 1.9665518618953464, 0.0029480342354130016])
    z_maxlen = 500
    sense_coord = np.array([0, 0, 0])
    zt_start = 0
    zt_end = np.pi

    Xt = []
    M, Mx, k, Mx_min = 0, 1, 0, len(Xt_1)
    H = [] # bins

    while M < Mx and M < Mx_min:
        sum_weights = sum(w for (_, w) in Xt_1)
        weights = [w / sum_weights for (_, w) in Xt_1]
        indices = np.random.choice(range(len(Xt_1)), len(Xt_1), p=weights)
        Xbar_t = [[Xt_1[i][0], Xt_1[i][1]] for i in indices]
        
        for _, (xt, wt) in enumerate(Xbar_t):
            xt = mm.sample_model_velocity(ut, xt, 1, alpha)
            wt = sm.likelihood_field_range_finder_model(zt, xt, m, theta, z_maxlen, sense_coord, zt_start, zt_end, transpose=True) 
            Xt.append([xt, wt])

            for h in H:
                if np.array_equal(h, xt):
                    break
            else:
                H.append(xt)
                k += 1

                if k > 1:
                    Z1 = np.random.normal(0, 1)
                    Mx = (k-1)/(2*epsilon) * (1- 2/(9*(k-1)) + np.sqrt(2/(9*(k-1))) * Z1-delta)**3
            M += 1
    return np.array(Xt)


def test_range_measurement(zt, Xbar_t, m , x):
    """
    Function that tests if the range measurement zt is consistent with the map m
    zt = sensor measurement
    Xbar_t = list of particles
    m = map
    x = threshold
    
    Returns True if the measurement is consistent, False otherwise
    """
    z_short, z_max, z_rand, z_hit, sigma_hit, lambda_short = 0.8157123986145881, 0.00235666025958796, 0.16552184295092348, 0.01640909817490046, 1.9665518618953464, 0.0029480342354130016
    zmax = 500
    p, q = 0, 0

    for m_, xt in enumerate(Xbar_t):
        ztk = zt[m_]
        ztk_star = sm.ray_casting(xt, m, 0, zmax, transpose=True)
        p  += z_short * sm.p_short(ztk, ztk_star, lambda_short)
        q  += z_hit * sm.p_hit(ztk, ztk_star, zmax, sigma_hit) + z_short * sm.p_short(ztk, ztk_star, lambda_short) + z_max * sm.p_max(ztk, zmax) + z_rand * sm.p_rand(ztk, zmax)

    if p/q <= x:
        return True
    else:
        return False
    