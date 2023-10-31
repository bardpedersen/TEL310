#!/usr/bin/env python3 

import sensor_model
import numpy as np

"""
Function for calculating p(zt|xt, m) at 21 different poses i the map, representing the robot's trajectory.
See images/task2.gif for the robot's trajectory in the map.
Her the green beam represents the robots front, theta from xt.
"""
def task2_likelihood():
    likelihood = []
    zt_start = 0
    zt_end = 2*np.pi
    z_maxlen = 500
    m = np.load('map/binary_image_cats.npy')
    z_hit = 0.8157123986145881
    z_short = 0.00235666025958796
    z_max = 0.16552184295092348
    z_rand =  0.01640909817490046
    sigma_hit = 1.9665518618953464
    lambda_short = 0.0029480342354130016
    theta = np.array([z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short])

    noise = np.random.normal(0, 2, size=8) 

    xt = np.array([30 * 7.2, 80 * 6, np.pi/2])
    zt = np.array([327.0, 203.0, 142, 99.0, 99.0, 149.0, 184.0, 327])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))
    
    xt = np.array([30 * 7.2, 75 * 6, np.pi/2])
    zt = np.array([297.0, 158.0, 209.00000000000003, 131.99999999999994, 132.00000000000006, 149.0, 156.99999999999997, 297.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))
    
    xt = np.array([30 * 7.2, 70 * 6, np.pi/2])
    zt = np.array([267.0, 107.00000000000001, 209.00000000000003, 165.0, 164.99999999999997, 150.00000000000003, 107.0, 267.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))
    
    xt = np.array([30 * 7.2, 65 * 6, np.pi/2])
    zt = np.array([237.0, 261.0, 209.00000000000003, 199.00000000000003, 198.99999999999997, 148.0, 330.0, 237.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))
    
    xt = np.array([30 * 7.2, 60 * 6, np.pi/2])
    zt = np.array([207.0, 261.0, 208.0, 232.00000000000006, 232.00000000000003, 148.0, 313.0, 207.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise    
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))
    
    xt = np.array([30 * 7.2, 55 * 6, np.pi/2])
    zt = np.array([177.0, 261.0, 66.99999999999999, 265.0, 265.0, 79.99999999999997, 297.0, 177.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))
    
    xt = np.array([30 * 7.2, 50 * 6, np.pi/2])
    zt = np.array([147.0, 261.0, 183.0, 299.0, 299.0, 389.0, 280.0, 147.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([33 * 7.2, 46 * 6, np.pi/3])
    zt = np.array([259.0, 123.99999999999999, 236.0, 114.99999999999999, 294.0, 95.0, 272.0, 259.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([36 * 7.2, 43 * 6, np.pi/6])
    zt = np.array([219.99999999999997, 248.99999999999997, 98.99999999999999, 248.0, 313.0, 325.0, 358.0, 219.99999999999997])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([40 * 7.2, 40 * 6, 0])
    zt = np.array([433.0, 196.99999999999997, 234.99999999999997, 105.99999999999999, 231.99999999999997, 338.0, 129.0, 433.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([41 * 7.2, 42 * 6, -np.pi/8])
    zt = np.array([334.0, 186.99999999999997, 243.99999999999997, 131.0, 284.0, 365.0, 93.00000000000001, 334.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([42 * 7.2, 44 * 6, -np.pi/4])
    zt = np.array([279.0, 420.99999999999994, 207.00000000000003, 268.0, 308.0, 343.0, 79.0, 279.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([43 * 7.2, 47 * 6, -np.pi/4])
    zt = np.array([83.00000000000003, 380.0000000000001, 211.0, 287.0, 315.0, 351.0, 59.000000000000036, 83.00000000000003])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([44 * 7.2, 49 * 6, -np.pi/8])
    zt = np.array([281.0, 192.0, 251.0, 187.0, 305.0, 337.0, 49.00000000000002, 280.99999999999994])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([45 * 7.2, 52 * 6, 0])
    zt = np.array([397.0, 206.00000000000003, 309.0, 346.0, 344.0, 29.999999999999996, 36.99999999999999, 397.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([53 * 7.2, 54 * 6, np.pi/12])
    zt = np.array([327.0, 183.00000000000003, 354.0, 376.00000000000006, 25.99999999999999, 244.99999999999994, 188.0, 326.99999999999994])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([60 * 7.2, 56 * 6, np.pi/6])
    zt = np.array([334.0, 177.0, 253.99999999999997, 60.99999999999997, 106.0, 151.0, 163.0, 334.00000000000006])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([68 * 7.2, 50 * 6, np.pi/4])
    zt = np.array([327.0, 110.00000000000001, 276.0, 125.99999999999999, 285.0, 133.99999999999997, 232.99999999999997, 327.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([75 * 7.2, 43 * 6, np.pi/4])
    zt = np.array([255.99999999999997, 247.00000000000003, 460.0, 249.99999999999997, 133.0, 135.99999999999997, 182.00000000000009, 256.00000000000006])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([83 * 7.2, 36 * 6, np.pi/4])
    zt = np.array([173.99999999999997, 98.00000000000001, 175.0, 500, 206.99999999999997, 232.00000000000003, 85.00000000000004, 173.99999999999997])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    zt[3] = 500
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    xt = np.array([90 * 7.2, 30 * 6, np.pi/4])
    zt = np.array([103.00000000000001, 170.0, 316.0, 140.99999999999997, 239.99999999999997, 54.00000000000003, 73.00000000000006, 103.0])
    noise = np.random.normal(0, 2, size=8) 
    zt += noise
    likelihood.append(sensor_model.beam_range_finder_model(zt, xt, m, theta, z_maxlen, zt_start, zt_end))

    for i in likelihood:
        print(i)

if __name__ == "__main__":
    task2_likelihood()