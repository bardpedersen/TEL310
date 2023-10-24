#!/usr/bin/env python3 

import math
import scipy.stats as stats
import numpy as np


"""
Algorithm: Probability Normal Distribution
Computing densities of a zero-centered normal distribution with variance c_2
Returns p(ztk|xt, m)
"""
def prob_normal_distribution(a, b, c_2):
    #return np.random.normal(a, math.sqrt(b_2))
    return (1/math.sqrt(2 * math.pi * c_2)) * math.exp(-0.5 * (((a-b) ** 2)/c_2))


"""
Algorithm: Gaussian distribution
Returns p(ztk|xt, m)
"""
def p_hit(ztk, ztk_star, zmax, sigma_hit):
    if 0 <= ztk <= zmax:
        N = prob_normal_distribution(ztk, ztk_star, sigma_hit**2)
        cdf = (stats.norm.cdf([0, zmax], loc=ztk_star, scale=sigma_hit))
        cdf = cdf[1] - cdf[0]
        return cdf**(-1) * N 
    
    return 0


"""
Algorithm: Exponential distribution
Returns p(ztk|xt, m)
"""
def p_short(ztk, ztk_star, lambda_short):
    if 0 <= ztk <= ztk_star:
        n = 1/(1-math.exp(-lambda_short*ztk_star))
        return n * lambda_short*math.exp(-lambda_short*ztk)
    
    return 0


"""
Algorithm: Uniform distribution
Returns p(ztk|xt, m)
"""
def p_max(ztk, zmax):
    if ztk == zmax:
        return 1
    
    return 0


"""
Algorithm: Uniform distribution
Returns p(ztk|xt, m)
"""
def p_rand(ztk, zmax):
    if 0 <= ztk <= zmax:
        return 1/zmax

    return 0
    

"""
xt =  [x, y, θ]T,  robot pose
m = map
theta_z = angle of the lidar reading
z_maxlen = max range of the lidar
transpose = if the map has switch x and y coordinates
"""
def ray_casting(xt, m, theta_z, z_maxlen, transpose=False):
    x, y, theta = xt # x, y, θ (robot pose)

    if transpose: # Transpose the map when x is y and y is x
        m = np.transpose(m)

    for i in range(int(z_maxlen)): # For every block in the map up to the max range
        m_x = x + i * np.cos(theta + theta_z) 
        m_y = y + i * -np.sin(theta + theta_z) # -sin is only nessasary when the map is wrong // else sin
        if m[round(m_x)][round(m_y)] == 1: # If the block is occupied
            return np.sqrt((m_x - x)**2 + (m_y - y)**2) # Return the distance to the block
        
    return z_maxlen # If no block is occupied return the max range


"""
zt = range scan in the range of [0;zmax]
xt = [x, y, θ]T, robot pose
m = map

theta = [z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short]
z_hit = weight for the hit probability
z_short = weight for the short probability
z_max = weight for the max probability
z_rand = weight for the random probability
sigma_hit = standard deviation for the hit probability
lambda_short = exponential decay rate for the short probability

The sum of the weights (z_..) should be 1

z_maxlen = max range of the lidar
zt_start = start angle of the lidar (often positiv)
zt_end = end angle of the lidar (often negativ)
"""
def beam_range_finder_model(zt, xt, m, Theta, z_maxlen, zt_start=0, zt_end=2*np.pi): #Start from left move to right
    z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short = Theta
    angle = np.linspace(zt_start, zt_end, len(zt)) # Each angle for each lidar reading
    q = 1.0

    for i, ztk in enumerate(zt):
        # Compute the expected measurement for the current pose and map
        ztk_star = ray_casting(xt, m, angle[i], z_maxlen, transpose=True)

        # Compute the probability of the actual measurement given the expected measurement
        p_hit_ = p_hit(ztk, ztk_star, z_max, sigma_hit)
        p_short_ = p_short(ztk, ztk_star, lambda_short)
        p_max_ = p_max(ztk, z_max)
        p_rand_ = p_rand(ztk, z_max)
        p = z_hit * p_hit_ + z_short * p_short_ + z_max * p_max_ + z_rand * p_rand_
        
        # Update the total probability
        q *= p
    
    return q


"""
"""
def learn_intrinsic_parameters(Z, X, m, Theta, z_maxlen=300, zt_start=0, zt_end=2*np.pi):
    z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short = Theta
    ei_hit = []
    ei_short = []
    ei_max = []
    ei_rand = []
    zi_star_list = []
    angle = np.linspace(zt_start, zt_end, len(Z))

    for i, zi in enumerate(Z): # Is zi one laser or multiple?
        zi_star = ray_casting(X[i], m, angle[i], z_maxlen, transpose=True)
        zi_star_list.append(zi_star)
        n = 1/(p_hit(zi, zi_star, z_max, sigma_hit) + p_short(zi, zi_star, lambda_short) + p_max(zi, z_max) + p_rand(zi, z_max))
        ei_hit.append(n * p_hit(zi, zi_star, z_max, sigma_hit))
        ei_short.append(n * p_short(zi, zi_star, lambda_short))
        ei_max.append(n * p_max(zi, z_max))
        ei_rand.append(n * p_rand(zi, z_max))

    z_hit = sum(ei_hit) / len(Z)
    z_short = sum(ei_short) / len(Z)
    z_max = sum(ei_max) / len(Z)
    z_rand = sum(ei_rand) / len(Z)
    sigma_hit = math.sqrt(sum(ei_hit * (zi - zi_star)**2 for zi, zi_star in zip(Z, zi_star_list)) / sum(ei_hit))
    lambda_short = sum(ei_short) / sum(ei_short * zi for zi in Z)
    return z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short


"""
"""
def likelihood_field_range_finder_model(zt, xt, m, Theta):
    z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short = Theta
    x, y, theta = xt

    q = 1
    xk_sens = None  # Vet ikke
    yk_sens = None  # vet ikke
    theta_k_sens = None  # vet ikke
    x_ = None # vet ikke
    y_ = None # vet ikke

    for ztk in zt:
        if ztk != z_max:
            xztk = x + xk_sens * np.cos(theta) - yk_sens * np.sin(theta) + ztk*np.cos(theta + theta_k_sens)
            yztk = y + yk_sens * np.cos(theta) + xk_sens * np.sin(theta) + ztk*np.sin(theta + theta_k_sens)

            dist = min(np.sqrt((xztk - x_)**2 + (yztk - y_)**2), z_max)

            prob_hit = stats.norm.pdf(dist, loc=0, scale=sigma_hit)
            q *= (z_hit * prob_hit + z_rand / z_max)
    
    return q

"""
cit = land marks
mj = coordinates for landmark 
"""
def landmark_model_known_correspondence(fit, cit, xt, m):
    x, y, theta = xt
    rit, thetait, sit = fit

    sigma_r = None
    sigma_theta = None
    sigma_s = None
    sj = None


    j = cit
    r_hat = np.sqrt((m[j][x] - x)**2 + (m[j][y] - y)**2)
    theta_hat = np.arctan2(m[j][y] - y, m[j][x] - x)
    return stats.norm.pdf(rit - r_hat, loc=0, scale=sigma_r) * stats.norm.pdf(thetait - theta_hat, loc=0, scale=sigma_theta) * stats.norm.pdf(sit - sj, loc=0, scale=sigma_s)
 

"""
"""
def  sample_landmark_model_known_correspondence(fit, cit, m):
    rit, thetait, sit = fit

    sigma_r = None
    sigma_theta = None

    j = cit
    gamma_hat = np.random.uniform(0, 2*np.pi)
    r_hat = rit + np.random.normal(0, sigma_r)
    theta_hat = thetait + np.random.normal(0, sigma_theta)
    x = m[j][x] + r_hat * np.cos(gamma_hat)
    y = m[j][y] + r_hat * np.sin(gamma_hat)
    theta = gamma_hat - np.pi - theta_hat
    return x, y, theta