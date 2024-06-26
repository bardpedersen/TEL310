#!/usr/bin/env python3 

import numpy as np
import math
import matplotlib.pyplot as plt
import Motion_models

"""
For testing the code
"""

# Testing the probability normal distribution
def test_prob_normal_distribution():
    x_plot = []
    y_plot = []
    b_2 = 4
    for i in np.linspace(-b_2, b_2, 1000000):
        y_plot.append(Motion_models.prob_normal_distribution(i, b_2))
        x_plot.append(i)
    plt.plot(x_plot, y_plot)
    plt.show()


# Testing the probability triangular distribution
def test_prob_triangular_distribution():
    x_plot = []
    y_plot = []
    b_2 = 4
    for i in np.linspace(-b_2, b_2, 1000000):
        y_plot.append(Motion_models.prob_triangular_distribution(i, b_2))
        x_plot.append(i)
    plt.plot(x_plot, y_plot)
    plt.show()


# Testing the sample normal distribution
def test_sample_normal_distribution():
    b_2 = 2
    x_plot = []
    for i in range(1000000):
        x_plot.append(Motion_models.sample_normal_distribution(b_2))
    plt.hist(x_plot, bins=10000)
    plt.show()


# Testing the sample triangular distribution
def test_sample_triangular_distribution():
    b_2 = 2
    x_plot = []
    for i in range(1000000):
        x_plot.append(Motion_models.sample_triangular_distribution(b_2))
    plt.hist(x_plot, bins=10000)
    plt.show()  


# Testing the motion model with velocity
def test_motion_model_velocity():
    test_xt = [1.8, 0.75, math.pi/4]  # Final position
    test_xt_1 = [0, 0, 0]  # Start position
    t = 2 # Time step
    test_ut = [1, math.pi/8] # Control input
    alpha = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15] # Motion error parameters
    print(Motion_models.motion_model_velocity(test_xt, test_ut, test_xt_1, t, alpha))


# Testing the motion model with odometry
def test_motion_model_odometry():
    test_xt = [1.8, 0.75, math.pi/4]  # Final position
    test_xt_1 = [0, 0, 0]  # Start position
    test_ut = [[0, 0, 0], [1.85, 0.77, 0.81]] # Control input [x_t_1, x_t] [x_t_1] = x, y, θ
    alpha = [0.05, 0.05, 0.05, 0.05]
    print(Motion_models.motion_model_odometry(test_xt, test_ut, test_xt_1, alpha))


# Testing the sample model with velocity
def test_sample_model_velocity():
    test_ut = [1, math.pi/8] # Control inpu
    test_xt = [1.8, 0.75, math.pi/4]  # Final position
    test_xt_1 = [0, 0, 0]  # Start position
    t = 2 # Time step
    alpha = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001] # Motion error parameters
    x_positions, y_positions, z_positions = zip(*(Motion_models.sample_model_velocity(test_ut, test_xt_1, t, alpha) for _ in range(100000)))
    plt.scatter(x_positions, y_positions, c='b', marker='o')
    plt.scatter(0,0, c='r', marker='o')
    plt.scatter(test_xt[0],test_xt[1], c='r', marker='o')
    plt.show()


# Testing the sample model with odometry
def test_sample_model_odometry():
    test_ut = [[0, 0, 0], [1.85, 0.77, 0.81]] # Control input [x_t_1, x_t] [x_t_1] = x, y, θ
    test_xt = [1.8, 0.75, math.pi/4]  # Final position
    test_xt_1 = [0, 0, 0]  # Start position
    alpha = [0.001, 0.001, 0.001, 0.001] # Motion error parameters
    x_positions, y_positions, z_positions = zip(*(Motion_models.sample_model_odometry(test_ut, test_xt_1, alpha) for _ in range(100000)))
    plt.scatter(x_positions, y_positions, c='b', marker='o')
    plt.scatter(0,0, c='r', marker='o')
    plt.scatter(test_xt[0],test_xt[1], c='r', marker='o')
    plt.show()
      

if __name__ == "__main__":
    test_prob_normal_distribution()
    test_prob_triangular_distribution()
    test_sample_normal_distribution()
    test_sample_triangular_distribution()
    test_motion_model_velocity()
    test_motion_model_odometry()
    test_sample_model_velocity()
    test_sample_model_odometry()
