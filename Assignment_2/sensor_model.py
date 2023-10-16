#!/usr/bin/env python3 

import math
import scipy.stats as stats

"""
"""
def prob_normal_distribution(a, b, c_2):
    #return np.random.normal(a, math.sqrt(b_2))
    return (1/math.sqrt(2 * math.pi * c_2)) * math.exp(-0.5 * (((a-b) ** 2)/c_2))

"""
"""
def p_hit(ztk, zmax, ztk_star, sigma_hit):
    if 0 <= ztk <= zmax:
        N = prob_normal_distribution(ztk, ztk_star, sigma_hit)
        cdf = (stats.norm.cdf([0, zmax], loc=ztk_star, scale=sigma_hit))**(-1)
        cdf = cdf[1] - cdf[0]
        return cdf * N 
    else:
        return 0

"""
"""
def p_short(ztk, ztk_star, lambda_short):
    if 0 <= ztk <= ztk_star:
        n = 1/(1-math.exp(-lambda_short*ztk_star))
        return n * lambda_short*math.exp(-lambda_short*ztk)
    else:
        return 0

"""
"""
def p_max(ztk, zmax):
    if ztk == zmax:
        return 1
    else:
        return 0
    
"""
"""
def p_rand(ztk, zmax):
    if 0 <= ztk <= zmax:
        return 1/zmax
    else:
        return 0

"""
zt = range scan in the range of [0;zmax]
xt = robot pose
m = map

def beam_range_finder_model(zt, xt, m): 
    q = 1.0
    for k in range(len(zt)):
        # Compute the expected measurement for the current pose and map
        zt_star = ray_casting(zt[k], xt, m)
        
        # Compute the probability of the actual measurement given the expected measurement
        p_hit = p_hit(zt[k], zt_star)
        p_short = p_short(zt[k], zt_star)
        p_max = p_max(zt[k], zt_star)
        p_rand = p_rand(zt[k], zt_star)
        p = z_hit * p_hit + z_short * p_short + z_max * p_max + z_rand * p_rand
        
        # Update the total probability
        q *= p
    
    return q
""" 