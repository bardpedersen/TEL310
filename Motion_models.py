#!/usr/bin/env python3 
import math
import random

"""

x_t = x(t)
u_t = u(t)
x_t_1 = x(t-1)

x_t=‚x′, y′, θ′]T
u_t=[v, w]T
x_t_1=[x, y, θ]T

x_ = x'
y_ = y'
theta_ = theta'

"""


# Algorithm: Motion Model Velocity
def motion_model_velocity(x_t, u_t, x_t_1, delta_t, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6):
    x_ = x_t[0]
    y_ = x_t[1]
    theta_ = x_t[2]
    v = u_t[0]
    w = u_t[1]
    x = x_t_1[0]
    y = x_t_1[1]
    theta = x_t_1[2]

    print((y - y_) * math.cos(theta) - (x - x_) * math.sin(theta))
    mu = 1 / 2 * ((x - x_) * math.cos(theta) + (y - y_) * math.sin(theta)) / ((y - y_) * math.cos(theta) - (x - x_) * math.sin(theta))
    
    x_star = (x + x_) / 2 + mu * (y - y_)
    y_star = (y + y_) / 2 + mu * (x_ - x)
    
    r_star = math.sqrt((x - x_star)**2 + (y - y_star)**2)
    delta_theta = math.atan2(y_ - y_star, x_ - x_star) - math.atan2(y - y_star, x - x_star)
    hat_omega = delta_theta / delta_t
    hat_v = hat_omega * r_star
    hat_gamma = (theta_ - theta)/delta_t - hat_omega
    
    p1 = prob_normal_distribution(v - hat_v, alpha1*v**2 + alpha2*w**2)
    p2 = prob_normal_distribution(w - hat_omega, alpha3*v**2 + alpha4*v**2)
    p3 = prob_normal_distribution(hat_gamma - theta, alpha5*v**2 + alpha6*v**2)
    return p1 * p2 * p3


# Algorithm: Probability Normal Distribution
def prob_normal_distribution(a, b_2):
    return (1 / math.sqrt(2 * math.pi * b_2)) * math.exp(-a**2 / (2 * b_2))


# Algorithm: Probability Triangular Distribution
def prob_triangular_distribution(a, b_2):
    return max(0, 1 / (math.sqrt(6)*math.sqrt(b_2)) - (abs(a)) / (6*(b_2)))


# Algorithm: Sample Model Velocity
def sample_model_velocity(u_t, x_t_1, delta_t, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6):
    v = u_t[0]
    w = u_t[1]
    x = x_t_1[0]
    y = x_t_1[1]
    theta = x_t_1[2]


    hat_v = v + sample_normal_distribution(alpha1 * v**2 + alpha2 * w**2)
    hat_w = w + sample_normal_distribution(alpha3 * v**2 + alpha4 * w**2)
    hat_gamma = sample_normal_distribution(alpha5 * v**2 + alpha6 * w**2)
    
    hat_r = hat_v / hat_w
    hat_theta = theta + hat_w * delta_t
    
    x_ = x - hat_r * (math.sin(theta) - math.sin(hat_theta))
    y_ = y + hat_r * (math.cos(theta) - math.cos(hat_theta))
    theta_ = hat_theta + hat_gamma * delta_t
    
    return [x_, y_, theta_]


# Algorithm: Sample Normal Distribution
def sample_normal_distribution(b_2):
    return sum(random.uniform(-math.sqrt(b_2), math.sqrt(b_2)) for _ in range(12)) / 2


# Algorithm: Sample Triangular Distribution
def sample_triangular_distribution(b_2):
    b = math.sqrt(b_2)
    return (math.sqrt(6) / 2) * [random.uniform(-b, b) + random.uniform(-b, b)]


# Algorithm: Motion Model Odometry
def motion_model_odometry(x_t, u_t, x_t_1, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6):
    hat_x_t_1 = u_t[0]
    hat_x_t = u_t[1]

    hat_x_ = hat_x_t[0]
    hat_y_ = hat_x_t[1]
    hat_theta_ = hat_x_t[2]
    hat_x = hat_x_t_1[0]
    hat_y = hat_x_t_1[1]
    hat_theta = hat_x_t_1[2]

    delta_rot1 = math.atan2(hat_y_ - hat_y, hat_x_ - hat_x) - hat_theta
    delta_trans = math.sqrt((hat_x_ - hat_x)**2 + (hat_y_ - hat_y)**2)
    delta_rot2 = hat_theta_ - hat_theta - delta_rot1
    
    x_ = x_t[0]
    y_ = x_t[1]
    theta_ = x_t[2]
    x = x_t_1[0]
    y = x_t_1[1]
    theta = x_t_1[2]

    hat_delta_rot1 = math.atan2(y_ - y, x_ - x) - theta
    hat_delta_trans = math.sqrt((x_ - x)**2 + (y_ - y)**2)
    hat_delta_rot2 = theta_ - theta - hat_delta_rot1

    p1 = prob_normal_distribution(delta_rot1 - hat_delta_rot1, alpha1*abs(delta_rot1) + alpha2*delta_trans)
    p2 = prob_normal_distribution(delta_trans - hat_delta_trans, alpha3*abs(delta_rot1) + alpha4*(abs(delta_rot1)*abs(delta_rot2)))
    p3 = prob_normal_distribution(delta_rot2 - hat_delta_rot2, alpha5*abs(delta_rot2) + alpha6*delta_trans)
    
    return p1 * p2 * p3


# Algorithm: Sample Model Odometry
def sample_model_odometry(u_t, x_t_1, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6):
    hat_x_t_1 = u_t[0]
    hat_x_t = u_t[1]

    hat_x_ = hat_x_t[0]
    hat_y_ = hat_x_t[1]
    hat_theta_ = hat_x_t[2]
    hat_x = hat_x_t_1[0]
    hat_y = hat_x_t_1[1]
    hat_theta = hat_x_t_1[2]

    x = x_t_1[0]
    y = x_t_1[1]
    theta = x_t_1[2]

    delta_rot1 = math.atan2(hat_y_ - hat_y, hat_x_ - hat_x) - hat_theta
    delta_trans = math.sqrt((hat_x_ - hat_x)**2 + (hat_y_ - hat_y)**2)
    delta_rot2 = hat_theta_ - hat_theta - delta_rot1


    hat_delta_rot1 = delta_rot1 - sample_normal_distribution(alpha1 * delta_rot1**2 + alpha2 * delta_trans**2)
    hat_delta_trans = delta_trans - sample_normal_distribution(alpha3 * delta_trans**2 + alpha4 * (delta_rot1**2 + delta_rot2**2))
    hat_delta_rot2 = delta_rot2 - sample_normal_distribution(alpha5 * delta_rot2**2 + alpha6 * delta_trans**2)
    
    x_ = x + hat_delta_trans * math.cos(theta + hat_delta_rot1)
    y_ = y + hat_delta_trans * math.sin(theta + hat_delta_rot1)
    theta_ = theta + hat_delta_rot1 + hat_delta_rot2
    
    return [x_, y_, theta_]

l1 = [1, 1, 1]
l2 = [2, 2, 2]
l3 = [3, 3, 3]
t= 1
alpha = [1, 1, 1, 1, 1, 1]
ut = [[1, 1, 1], [1, 1, 1]]



print(motion_model_velocity(l1, l2, l3, t, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5]))
print(sample_model_velocity(l1, l2, t, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5]))
print(motion_model_odometry(l1, ut, l3, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5]))
print(sample_model_odometry(ut, l2, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5]))