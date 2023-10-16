#!/usr/bin/env/python3

import sensor_model


def test_p_hit():
    ztk = 1
    zmax = 10
    ztk_star = 1
    sigma_hit = 1
    print(sensor_model.p_hit(ztk, zmax, ztk_star, sigma_hit))

def test_p_short():
    ztk = 1
    ztk_star = 1
    lambda_short = 1
    print(sensor_model.p_short(ztk, ztk_star, lambda_short))

def test_p_max():
    ztk = 10
    zmax = 10
    print(sensor_model.p_max(ztk, zmax))

def test_p_rand():
    ztk = 1
    zmax = 10
    print(sensor_model.p_rand(ztk, zmax))


test_p_hit()
test_p_short()
test_p_max()
test_p_rand()
