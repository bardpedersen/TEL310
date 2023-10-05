#!/usr/bin/env python3 
"""
Porblems when alpha values are too small
"""
import math
import random
import matplotlib.pyplot as plt


"""
Algorithm: Motion Model Velocity

x_t = [x', y', θ']T , this is the final (hypotetical) position
u_t = [v, w]T , this is the control input
x_t_1 = [x, y, θ]T , this is the initial position
delta_t is the time step or the time interval between the two positions 
alpha1 to alpha6 are robot-specific motion error parameters

It returns the probability of the final position given the initial position, the control input, the time step and the motion error parameters.
"""
def motion_model_velocity(x_t, u_t, x_t_1, delta_t, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6):
    x_ = x_t[0] # x'
    y_ = x_t[1] # y'
    theta_ = x_t[2] # θ'
    v = u_t[0] # v
    w = u_t[1] # w
    x = x_t_1[0] # x
    y = x_t_1[1] # y
    theta = x_t_1[2] # θ

    mu = 1 / 2 * ((x - x_) * math.cos(theta) + (y - y_) * math.sin(theta)) / ((y - y_) * math.cos(theta) - (x - x_) * math.sin(theta))
    
    # Center of rotation obtained from the positions
    x_star = (x + x_) / 2 + mu * (y - y_)
    y_star = (y + y_) / 2 + mu * (x_ - x)
    r_star = math.sqrt((x - x_star)**2 + (y - y_star)**2)

    delta_theta = math.atan2(y_ - y_star, x_ - x_star) - math.atan2(y - y_star, x - x_star)

    # Velocities obtained from the positions
    hat_omega = delta_theta / delta_t
    hat_v = hat_omega * r_star
    hat_gamma = (theta_ - theta)/delta_t - hat_omega
    
    # prob(error, variance)
    p1 = prob_triangular_distribution(v - hat_v, alpha1*v**2 + alpha2*w**2)
    p2 = prob_triangular_distribution(w - hat_omega, alpha3*v**2 + alpha4*w**2)
    p3 = prob_triangular_distribution(hat_gamma - theta, alpha5*v**2 + alpha6*w**2)
    return p1 * p2 * p3 # p(x_t|x_t_1, u_t)


"""
Algorithm: Probability Normal Distribution
Computing densities of a zero-centered normal distribution with variance b2
"""
def prob_normal_distribution(a, b_2):
    return (1 / math.sqrt(2 * math.pi * b_2)) * math.exp(-a**2 / (2 * b_2))


"""
Algorithm: Probability Triangular Distribution
Computing densities of a zero-centered triangular distribution with variance b2
"""
def prob_triangular_distribution(a, b_2):
    return max(0, 1 / (math.sqrt(6)*math.sqrt(b_2)) - (abs(a)) / (6*(b_2)))


"""
Algorithm: Sample Model Velocity

u_t = [v, w]T , this is the control input
x_t_1 = [x, y, θ]T , this is the initial position
delta_t is the time step or the time interval between the two positions
alpha1 to alpha6 are the parameters of the motion noise

It returns a generated position that is a posibiltiy with the given initial position, the control input, the time step and the motion error parameters.
"""
def sample_model_velocity(u_t, x_t_1, delta_t, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6):
    v = u_t[0] # v
    w = u_t[1] # w
    x = x_t_1[0] # x
    y = x_t_1[1] # y
    theta = x_t_1[2] # θ

    # Sample noise from a normal distribution with parameters influenced by velocity and acceleration
    hat_v = v + sample_normal_distribution(alpha1 * v**2 + alpha2 * w**2) 
    hat_w = w + sample_normal_distribution(alpha3 * v**2 + alpha4 * w**2)
    hat_gamma = sample_normal_distribution(alpha5 * v**2 + alpha6 * w**2)
    
    # Calculate the predicted pose values based on sampled noise
    hat_r = hat_v / hat_w
    hat_theta = theta + hat_w * delta_t
    
    # Update the estimated position using kinematic equations
    x_ = x - hat_r * (math.sin(theta) - math.sin(hat_theta))
    y_ = y + hat_r * (math.cos(theta) - math.cos(hat_theta))
    theta_ = hat_theta + hat_gamma * delta_t
    return [x_, y_, theta_] # [x', y', θ']T


"""
Algorithm: Sample Normal Distribution
Generates sampling from (approximate) normal distributions with zero mean and variance b2
"""
def sample_normal_distribution(b_2):
    return sum(random.uniform(-math.sqrt(b_2), math.sqrt(b_2)) for _ in range(12)) / 2


"""
Algorithm: Sample Triangular Distribution
Generates sampling from (approximate) triangular distributions with zero mean and variance b2
"""
def sample_triangular_distribution(b_2):
    b = math.sqrt(b_2)
    return (math.sqrt(6) / 2) * (random.uniform(-b, b) + random.uniform(-b, b))


"""
Algorithm: Motion Model Odometry

x_t = [x', y', θ']T , this is the final (hypotetical) position
u_t = [[x_t_1], [x_t]] , this is the control input, 
where [x_t_1] = x̂, ŷ, θ̂ is the start position for the odom and 
[x_t] = x̂', ŷ', θ̂' is the end position for the odom

x_t_1 = [x, y, θ]T , this is the initial position
alpha1 to alpha6 are robot-specific motion error parameters

It returns the probability of the final position given the initial position, the control input and the motion error parameters.
p(x_t|x_t_1, u_t)
"""
def motion_model_odometry(x_t, u_t, x_t_1, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6):
    hat_x_t_1 = u_t[0] # [x_t_1] = x̂, ŷ, θ̂
    hat_x_t = u_t[1] # [x_t] = x̂', ŷ', θ̂'

    hat_x_ = hat_x_t[0] # x̂'
    hat_y_ = hat_x_t[1] # ŷ'
    hat_theta_ = hat_x_t[2] # θ̂'
    hat_x = hat_x_t_1[0] # x̂
    hat_y = hat_x_t_1[1] # ŷ
    hat_theta = hat_x_t_1[2] # θ̂

    # Calculate the differences in orientation and translation based on estimated states
    delta_rot1 = math.atan2(hat_y_ - hat_y, hat_x_ - hat_x) - hat_theta
    delta_trans = math.sqrt((hat_x_ - hat_x)**2 + (hat_y_ - hat_y)**2)
    delta_rot2 = hat_theta_ - hat_theta - delta_rot1
    
    x_ = x_t[0] # x'
    y_ = x_t[1] # y'
    theta_ = x_t[2] # θ'
    x = x_t_1[0] # x   
    y = x_t_1[1] # y
    theta = x_t_1[2] # θ

    # Calculate the differences in orientation and translation based on true states
    hat_delta_rot1 = math.atan2(y_ - y, x_ - x) - theta
    hat_delta_trans = math.sqrt((x_ - x)**2 + (y_ - y)**2)
    hat_delta_rot2 = theta_ - theta - hat_delta_rot1

    # Calculate probabilities using the difference values and noise parameters
    p1 = prob_normal_distribution(delta_rot1 - hat_delta_rot1, alpha1*hat_delta_rot1**2 + alpha2*hat_delta_trans**2)
    p2 = prob_normal_distribution(delta_trans - hat_delta_trans, alpha3*hat_delta_trans**2 + alpha4*hat_delta_rot1**2 + alpha4*hat_delta_rot2**2)
    p3 = prob_normal_distribution(delta_rot2 - hat_delta_rot2, alpha5*hat_delta_rot2**2 + alpha6*hat_delta_trans**2)  
    return p1 * p2 * p3 # p(x_t|x_t_1, u_t)


"""
Algorithm: Sample Model Odometry

u_t = [[x_t_1], [x_t]] , this is the control input, 
where [x_t_1] = x̂, ŷ, θ̂ is the start position for the odom and
[x_t] = x̂', ŷ', θ̂' is the end position for the odom

x_t_1 = [x, y, θ]T , this is the initial position
alpha1 to alpha6 are robot-specific motion error parameters
"""
def sample_model_odometry(u_t, x_t_1, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6):
    hat_x_t_1 = u_t[0] # [x_t_1] = x̂, ŷ, θ̂
    hat_x_t = u_t[1] # [x_t] = x̂', ŷ', θ̂'
    hat_x_ = hat_x_t[0] # x̂'
    hat_y_ = hat_x_t[1] # ŷ'
    hat_theta_ = hat_x_t[2] # θ̂'
    hat_x = hat_x_t_1[0] # x̂
    hat_y = hat_x_t_1[1] # ŷ
    hat_theta = hat_x_t_1[2] # θ̂
    x = x_t_1[0] # x
    y = x_t_1[1] # y
    theta = x_t_1[2] # θ

    # Calculate differences in orientation and translation based on estimated states
    delta_rot1 = math.atan2(hat_y_ - hat_y, hat_x_ - hat_x) - hat_theta
    delta_trans = math.sqrt((hat_x_ - hat_x)**2 + (hat_y_ - hat_y)**2)
    delta_rot2 = hat_theta_ - hat_theta - delta_rot1

    # Introduce noise to the differences using sampled values from a normal distribution
    hat_delta_rot1 = delta_rot1 - sample_normal_distribution(alpha1 * delta_rot1**2 + alpha2 * delta_trans**2)
    hat_delta_trans = delta_trans - sample_normal_distribution(alpha3 * delta_trans**2 + alpha4 * (delta_rot1**2 + delta_rot2**2))
    hat_delta_rot2 = delta_rot2 - sample_normal_distribution(alpha5 * delta_rot2**2 + alpha6 * delta_trans**2)
    
    # Update the estimated state with the noisy differences
    x_ = x + hat_delta_trans * math.cos(theta + hat_delta_rot1)
    y_ = y + hat_delta_trans * math.sin(theta + hat_delta_rot1)
    theta_ = theta + hat_delta_rot1 + hat_delta_rot2
    return [x_, y_, theta_] # [x', y', θ']T


"""
For testing the code
"""
if __name__ == '__main__':
    test_xt = [1.8, 0.75, math.pi/4]  # Final position
    test_xt_1 = [0, 0, 0]  # Start position
    
    # Testing the motion model with velocity
    t = 2 # Time step
    test_ut = [1, math.pi/8] # Control input
    alpha = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15] # Motion error parameters
    alpha = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001] # Motion error parameters
    print(motion_model_velocity(test_xt, test_ut, test_xt_1, t, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5]))

    # Testing the sample model with velocity
    x_positions, y_positions, z_positions = zip(*(sample_model_velocity(test_ut, test_xt_1, t, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5]) for _ in range(1000)))
    plt.scatter(x_positions, y_positions, c='b', marker='o')
    plt.scatter(0,0, c='r', marker='o')
    plt.scatter(test_xt[0],test_xt[1], c='r', marker='o')
    plt.show()

    # Testing the motion model with odometry
    test_ut = [[0, 0, 0], [1.85, 0.77, 0.81]] # Control input [x_t_1, x_t] [x_t_1] = x, y, θ
    #alpha = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    print(motion_model_odometry(test_xt, test_ut, test_xt_1, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5]))

    # Testing the sample model with odometry
    x_positions, y_positions, z_positions = zip(*(sample_model_odometry(test_ut, test_xt_1, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5]) for _ in range(1000)))
    plt.scatter(x_positions, y_positions, c='b', marker='o')
    plt.scatter(0,0, c='r', marker='o')
    plt.scatter(test_xt[0],test_xt[1], c='r', marker='o')
    plt.show()
