import random
import math
import numpy as np
import timeit

# Function to generate normal samples by summing 12 uniform samples
# mu = mean, where the middle is
# sigma = variance, how spread out the distribution is one veriance to each side of the mu is 68% of the data

def  normal_distribution_N(mu, sigma):
    total = sum([random.uniform(0, 1) for _ in range(12)])
    return mu + math.sqrt(sigma) * (total - 6)

# Function to generate normal samples using rejection sampling
def normal_rejection(mu, sigma_squared):
    while True:
        x = random.uniform(0, 1)
        y = random.uniform(0, math.exp(-0.5))
        z = math.sqrt(-2 * math.log(y))
        if random.uniform(0, 1) <= math.exp(-0.5 * (z - 1)**2):
            return mu + math.sqrt(sigma_squared) * z

# Function to generate normal samples using Box-Muller transformation
def normal_box_muller(mu, sigma_squared):
    u1 = random.uniform(0, 1)
    u2 = random.uniform(0, 1)
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mu + math.sqrt(sigma_squared) * z

# Benchmark the execution times using timeit
mu = 0
sigma_squared = 1
num_samples = 100

# Using numpy.random.normal
def numpy_random_normal():
    np.random.normal(mu, math.sqrt(sigma_squared), num_samples)

# Time the functions
"""
sum_time = timeit.timeit(lambda: [normal_sum(mu, sigma_squared) for _ in range(num_samples)], number=100)
rejection_time = timeit.timeit(lambda: [normal_rejection(mu, sigma_squared) for _ in range(num_samples)], number=100)
box_muller_time = timeit.timeit(lambda: [normal_box_muller(mu, sigma_squared) for _ in range(num_samples)], number=100)
numpy_time = timeit.timeit(numpy_random_normal, number=100)

# Print the execution times
if __name__ == "__main__":
    print(normal_sum(mu, sigma_squared))
    print(normal_rejection(mu, sigma_squared))
    print(normal_box_muller(mu, sigma_squared))
    print(numpy_random_normal())
    print(f"Summing 12 Uniform Samples: {sum_time:.6f} seconds")
    print(f"Rejection Sampling: {rejection_time:.6f} seconds")
    print(f"Box-Muller Transformation: {box_muller_time:.6f} seconds")
    print(f"numpy.random.normal: {numpy_time:.6f} seconds")
"""