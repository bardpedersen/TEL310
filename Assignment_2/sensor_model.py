#!/usr/bin/env python3 

import math
import scipy.stats as stats
import numpy as np


"""
Algorithm: Probability Normal Distribution
a = value
b = mean
c_2 = variance

Returns p(ztk|xt, m)
"""
def prob_normal_distribution(a, b, c_2):
    #return np.random.normal(a, math.sqrt(b_2))
    if c_2 == 0:
        return 0
    return (1/math.sqrt(2 * math.pi * c_2)) * math.exp(-0.5 * (((a-b) ** 2)/c_2))


"""
Algorithm: Gaussian distribution
ztk = lidar reading
ztk_star = calculated lidar reading
zmax = max range of the lidar
sigma_hit = standard deviation for the hit probability

Returns p(ztk|xt, m)
"""
def p_hit(ztk, ztk_star, zmax, sigma_hit):
    if 0 <= ztk <= zmax:
        normalize_hit = 0
        for j in range(int(zmax)):
            normalize_hit += prob_normal_distribution(j, ztk_star, sigma_hit**2)
    
        N = 1 / normalize_hit
        return N * prob_normal_distribution(ztk, ztk_star, sigma_hit**2)
    
    return 0


"""
Algorithm: Exponential distribution
ztk = lidar reading
ztk_star = calculated lidar reading
lambda_short = exponential decay rate

Returns p(ztk|xt, m)
"""
def p_short(ztk, ztk_star, lambda_short):
    if 0 <= ztk <= ztk_star:
        n = 1/(1-math.exp(-lambda_short*ztk_star))
        return n * lambda_short*math.exp(-lambda_short*ztk)
    
    return 0


"""
Algorithm: Uniform distribution
ztk = lidar reading
zmax = max range of the lidar

Returns p(ztk|xt, m)
"""
def p_max(ztk, zmax):
    if ztk == zmax:
        return 1
    
    return 0


"""
Algorithm: Uniform distribution
ztk = lidar reading
zmax = max range of the lidar

Returns p(ztk|xt, m)
"""
def p_rand(ztk, zmax):
    if 0 <= ztk <= zmax:
        return 1/zmax

    return 0
    

"""
Algorithm: Ray casting
xt =  [x, y, θ]T,  robot pose
x = x position of robot
y = y position of robot
θ = orientation of robot

m = map

theta_z = angle of the lidar reading
z_maxlen = max range of the lidar
transpose = if the map has switch x and y coordinates

Returns the calculated distance to the first occupied block in the map or the max range
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
Algorithm: Beam range finder model
zt = range scans
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

Returns the probability of the lidar reading given the robot pose and the map

Dos not work
"""
def beam_range_finder_model(zt, xt, m, Theta, z_maxlen, zt_start=0, zt_end=2*np.pi, transpose=True): #Start from left move to right
    z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short = Theta
    angle = np.linspace(zt_start, zt_end, len(zt)) # Each angle for each lidar reading
    q = 1.0

    for i, ztk in enumerate(zt):
        # Compute the expected measurement for the current pose and map
        ztk_star = ray_casting(xt, m, angle[i], z_maxlen, transpose)

        # Compute the probability of the actual measurement given the expected measurement
        p_hit_ = p_hit(ztk, ztk_star, z_maxlen, sigma_hit)
        p_short_ = p_short(ztk, ztk_star, lambda_short)
        p_max_ = p_max(ztk, z_maxlen)
        p_rand_ = p_rand(ztk, z_maxlen)
        p = z_hit * p_hit_ + z_short * p_short_ + z_max * p_max_ + z_rand * p_rand_
        # Update the total probability
        q *= p
        
    return q


"""
Algorithm: Learn intrinsic parameters
Z = data set where each zi is an actual measurement
X = data set where each xi is a robot pose
m = map
Theta = [z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short]

z_hit = weight for the hit probability
z_short = weight for the short probability
z_max = weight for the max probability
z_rand = weight for the random probability
sigma_hit = standard deviation for the hit probability
lambda_short = exponential decay rate for the short probability
The sum of the weights (z_..) should be 1

z_maxlen = max range of the lidar
angle = angle of the lidar with respect to the robot

Returns the learned intrinsic parameters

Todo add so Z contains multple readings
zt_start=0, zt_end=2*np.pi
"""
def learn_intrinsic_parameters(Z, X, m, Theta, z_maxlen=500, angle=0):
    cur = Theta 
    z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short = cur
    convergence = False

    while convergence == False:
        ei_hit = []
        ei_short = []
        ei_max = []
        ei_rand = []
        zi_star_list = [] 
        prev = cur
        
        for i, zi in enumerate(Z): # Zi one laser reading, Xi is one postion
            zi_star = ray_casting(X[i], m, angle, z_maxlen, transpose=True)
            zi_star_list.append(zi_star)
            n = 1/(p_hit(zi, zi_star, z_maxlen, sigma_hit) + p_short(zi, zi_star, lambda_short) + p_max(zi, z_maxlen) + p_rand(zi, z_maxlen))
            ei_hit.append(n * p_hit(zi, zi_star, z_maxlen, sigma_hit))
            ei_short.append(n * p_short(zi, zi_star, lambda_short))
            ei_max.append(n * p_max(zi, z_maxlen))
            ei_rand.append(n * p_rand(zi, z_maxlen))

        z_hit = sum(ei_hit) / len(Z)
        z_short = sum(ei_short) / len(Z)
        z_max = sum(ei_max) / len(Z)
        z_rand = sum(ei_rand) / len(Z)
        sigma_hit = math.sqrt(sum(ei_hit[i] * (Z[i] - zi_star_list[i])**2 for i in range(len((Z)))) / sum(ei_hit))
        lambda_short = sum(ei_short) / sum(ei_short[i] * Z[i] for i in range(len((Z))))

        # Check if the parameters have converged
        cur = [z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short]
        convergence = np.allclose(prev, cur, atol=0.01)

    return z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short


"""
Algorithm: Likelihood field range finder model
zt = multiple range scans

xt = [x, y, θ]T, robot pose
x = x position of robot
y = y position of robot
θ = orientation of robot

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

sense_coord = [Xk_sense, Yk_sense, Thetak_sense] sensor location in the robot coordinate system

Returns the probability of the reading distance given the robot pose and the map
"""
def likelihood_field_range_finder_model(zt, xt, m, Theta, z_maxlen, sense_coord, zt_start, zt_end, transpose=False):
    x, y, theta = xt # x, y, θ (robot pose)
    z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short = Theta
    xk_sens, yk_sens, theta_k_sens = sense_coord  # Sensor location in the robot coordinate system    
    angle = np.linspace(zt_start, zt_end, len(zt))
    q = 1

    if transpose:
        m = np.transpose(m)

    blocked_list = [] # List of all blocked cells in the map
    for x_i in range(m.shape[0]):
        for y_i in range(m.shape[1]):
            if m[x_i][y_i] == 1:
                blocked_list.append([x_i, y_i])

    for i, ztk in enumerate(zt):
        if ztk != z_maxlen:
            xztk = x + xk_sens * np.cos(theta) - yk_sens * np.sin(theta) + ztk*np.cos(theta + theta_k_sens + angle[i])
            yztk = y + yk_sens * np.cos(theta) + xk_sens * np.sin(theta) - ztk*np.sin(theta + theta_k_sens + angle[i]) # -sin is only nessasary when the map is wrong // else sin
            list_dist = []
            for x_, y_ in blocked_list:
                list_dist.append(np.sqrt((xztk - x_)**2 + (yztk - y_)**2)) # Distance to all blocked cells in the map

            dist = min(list_dist)
            prob_hit = stats.norm.pdf(dist, loc=0, scale=sigma_hit)
            q *= (z_hit * prob_hit + z_rand / z_maxlen)
    
    return q


"""
Algorithm: Landmark model with known correspondence
fit = [rit, thetait, sit]T, the landmark measurement
rit = distance
thetait = bearing
sit = signature

cit = [jx, jy, ji]T, the landmark identity
jx = x position of landmark
jy = y position of landmark
ji = landmark identity

xt = [x, y, θ]T, robot pose
x = x position of robot
y = y position of robot
θ = orientation of robot

sigma = [sigma_r, sigma_theta, sigma_s]T, noise
sigma_r = noise in distance
sigma_theta = noise in bearing
sigma_s = noise in signature

Returns the probability of the landmark measurement given the robot pose and the map
"""
def landmark_model_known_correspondence(fit, cit, xt, sigma):
    rit, thetait, sit = fit
    j_x, j_y, sj = cit
    x, y, theta = xt
    sigma_r, sigma_theta, sigma_s = sigma

    r_hat = np.sqrt((j_x - x)**2 + (j_y - y)**2)
    theta_hat = np.arctan2(j_y - y, j_x - x) - theta
    return stats.norm.pdf(rit - r_hat, loc=0, scale=sigma_r) * stats.norm.pdf(thetait - theta_hat, loc=0, scale=sigma_theta) * stats.norm.pdf(sit - sj, loc=0, scale=sigma_s)
 

"""
Algorithm: Landmark model with unknown correspondence
fit = [rit, thetait, sit]T, the landmark measurement
rit = distance
thetait = bearing
sit = signature

cit = [jx, jy, ji]T, the landmark identity
jx = x position of landmark
jy = y position of landmark
ji = landmark identity

m = map

sigma = [sigma_r, sigma_theta]T, noise

Returns the pose of the robot
"""
def  sample_landmark_model_known_correspondence(fit, cit, m, sigma):
    rit, thetait, sit = fit # rit = distance, thetait = bearing, sit = signature
    sigma_r, sigma_theta = sigma # noise 

    j = cit # landmark identity
    j_x, j_y, j_i = j
    gamma_hat = np.random.uniform(0, 2*np.pi)
    r_hat = rit + np.random.normal(0, sigma_r)
    theta_hat = thetait + np.random.normal(0, sigma_theta)
    x = j_x + r_hat * np.cos(gamma_hat)
    y = j_y + r_hat * -np.sin(gamma_hat)
    theta = gamma_hat - np.pi - theta_hat


    # Only send data if the robot is in a free space, within the map, and no obsticle in the way
    m = np.transpose(m)
    if not (0 < x < m.shape[0] and 0 < y < m.shape[1]): # Check if the robot is within the map
        return 0, 0, 0
    
    if m[int(x)][int(y)] == 1: # Check if the robot is in a free space
        return 0, 0, 0
    
    # check if not obsticle in the way
    for i in range(int(r_hat)): 
        m_x = j_x + i * np.cos(gamma_hat) 
        m_y = j_y + i * -np.sin(gamma_hat)
        if 0 <= m_x < m.shape[0]-1 and 0 <= m_y < m.shape[1]-1:
            if m[round(m_x)][round(m_y)] == 1:
                return 0,0,0

    return x, y, theta
