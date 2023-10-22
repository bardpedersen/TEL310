#!/usr/bin/env/python3

import sensor_model
import matplotlib.pyplot as plt
import numpy as np

def test_p_hit():
    list_av_values = []
    zmax = 10
    z = np.linspace(0, zmax, 1000)
    ztk_star = 5
    sigma_hit = 1
    for ztk in z:
        list_av_values.append(sensor_model.p_hit(ztk, ztk_star, zmax, sigma_hit))
    plt.plot(z, list_av_values)
    plt.show()

def test_p_short():
    list_av_values = []
    zmax = 10
    z = np.linspace(0, zmax, 1000)
    ztk_star = 5
    lambda_short = 1
    for ztk in z:
        list_av_values.append(sensor_model.p_short(ztk, ztk_star, lambda_short))
    plt.plot(z, list_av_values)
    plt.show()

def test_p_max():
    list_av_values = []
    zmax = 10
    z = np.linspace(0, zmax, 1000)
    for ztk in z:
        list_av_values.append(sensor_model.p_max(ztk, zmax))
    plt.plot(z, list_av_values)
    plt.show()

def test_p_rand():
    list_av_values = []
    zmax = 10
    z = np.linspace(0, zmax, 1000)
    for ztk in z:
        list_av_values.append(sensor_model.p_rand(ztk, zmax))
    plt.plot(z, list_av_values)
    plt.show()

def test_all_p():
    list_av_values = []
    zmax = 10
    z = np.linspace(0, zmax, 1000)
    ztk_star = 5
    sigma_hit = 1
    lambda_short = 1
    for ztk in z:
        value = 0
        value += (sensor_model.p_hit(ztk, ztk_star, zmax, sigma_hit))
        value += (sensor_model.p_short(ztk, ztk_star, lambda_short))
        value += (sensor_model.p_max(ztk, zmax))
        value += (sensor_model.p_rand(ztk, zmax))
        #print(value)
        list_av_values.append(value/4)
    plt.plot(z, list_av_values)
    plt.show()  

test_p_hit()
test_p_short()
test_p_max()
test_p_rand()
test_all_p()
