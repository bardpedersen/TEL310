#!/usr/bin/env python3 
import numpy as np


def EKF_SLAM_known_correspondences(μt_1, Σt_1, ut, zt, ct, number_of_landmarks, sigma):
    """
    Algorithm for calculating the EKF SLAM with known correspondences.

    μt_1 = [xt_1, yt_1, θt_1] is the mean of the robot pose at time t-1
    Σt_1 is the covariance of the robot pose at time t-1 (3x3)
    ut = [vt, wt] is the control input
    zt = {zt1,zt2,...} ranges to landmarks, where each zti = [rti, phiti, sti] is the range, bearing, and signature of the landmark
    ct = {ct1,ct2,...} is the correspondence of each landmark
    number_of_landmarks is the number of landmarks in the map
    sigma = [sigma_r, sigma_phi, sigma_s] is the standard deviation of the measurements    

    Returns:
    μt is the mean of the robot pose at time t
    Σt is the covariance of the robot pose at time t (3x3)
    """
    x, y, theta = μt_1
    vt, wt = ut
    sigma_r, sigma_phi, sigma_s = sigma
    N = number_of_landmarks
    dt = 1 # time step
    Rt = np.diagflat(np.array([5.0, 5.0, 100.0])) ** 2
    print(Rt)

    Fx = np.block([np.eye(3),np.zeros((3, 3*N))])

    μ_bar_t = μt_1 + Fx.T @ ([[-vt/wt*np.sin(theta) + vt/wt*np.sin(theta + wt*dt)],
                            [vt/wt*np.cos(theta) - vt/wt*np.cos(theta + wt*dt)],
                            [wt*dt]]) # 3+3N x 1

    Gt = np.eye(3+3*N) + Fx.T @ ([[0, 0, -vt/wt*np.cos(theta) + vt/wt*np.cos(theta + wt*dt)],
                                [0, 0, -vt/wt*np.sin(theta) + vt/wt*np.sin(theta + wt*dt)],
                                [0, 0, 0]]) @ Fx # 3+3N x 3+3N
    
    Σ_bar_t = Gt @ Σt_1 @ Gt.T + Fx.T @ Rt @ Fx 

    Qt = ([[sigma_r**2, 0, 0],
            [0, sigma_phi**2, 0],
            [0, 0, sigma_s**2]]) # 3x3

    for i in range(zt):
        rit, phit, sit = zt[i]
        j = ct[i]
        if True:
            μ_bar_j = ([[μ_bar_t[0] + rit*np.cos(phit + μ_bar_t[2])],
                        [μ_bar_t[1] + rit*np.sin(phit + μ_bar_t[2])],
                        [sit]])

        delta = np.array([[μ_bar_j[0] - μ_bar_t[0]],
                        [μ_bar_j[1] - μ_bar_t[1]]])
        
        q = delta.T @ delta

        z_hat_it = np.array([[np.sqrt(q)],
                            [np.arctan2(delta[1], delta[0]) - μ_bar_t[2]],
                            [μ_bar_j[2]]])
        
        Fxj = np.block([[np.eye(3), np.zeros((3, 3*j-3)), np.zeros(3, 3), np.zeros((3, 3*(N-j)))],
                        [np.zeros((3, 3)), np.zeros((3, 3*j-3)), np.eye(3), np.zeros((3, 3*(N-j)))]])
        
        Hit = (np.array([[-np.sqrt(q)*delta[0], -np.sqrt(q)*delta[1], 0, np.sqrt(q)*delta[0], np.sqrt(q)*delta[1], 0],
                        [delta[1], -delta[0], -q, -delta[1], delta[0], 0],
                        [0, 0, 0, 0, 0, q]]) / q) @ Fxj
        
        Kit = Σ_bar_t @ Hit.T @ np.linalg.inv(Hit @ Σ_bar_t @ Hit.T + Qt)
        μ_bar_t = μ_bar_t + Kit @ (zt[i] - z_hat_it) # 3x1
        Σ_bar_t = (np.eye(3+3*N) - Kit @ Hit) @ Σ_bar_t # 3x3

    return μ_bar_t, Σ_bar_t


def EKF_SLAM(μt_1, Σt_1, ut, zt, Nt_1, sigma):
    x, y, theta = μt_1
    vt, wt = ut
    sigma_r, sigma_phi, sigma_s = sigma

    dt = 0.1 #######
    N = 1 #########

    Fx = np.block([np.eye(3),np.zeros((3, 3*N))])
    μ_bar_t = μt_1 + Fx.T @ ([[-vt/wt*np.sin(theta) + vt/wt*np.sin(theta + wt*dt)],
                            [vt/wt*np.cos(theta) - vt/wt*np.cos(theta + wt*dt)],
                            [wt*dt]])
    
    Gt = np.eye(3) + Fx.T @ ([[0, 0, -vt/wt*np.cos(theta) + vt/wt*np.cos(theta + wt*dt)],
                                [0, 0, -vt/wt*np.sin(theta) + vt/wt*np.sin(theta + wt*dt)],
                                [0, 0, 0]]) @ Fx
    
    Σ_bar_t = Gt @ Σt_1 @ Gt.T + Fx.T @ Rt @ Fx

    Qt = ([[sigma_r**2, 0, 0],
            [0, sigma_phi**2, 0],
            [0, 0, sigma_s**2]])
    
    for i in range(zt):
        rit, phit, sit = zt[i]

        μ_bar_Nt1 = ([[μ_bar_t[0] + rit*np.cos(phit + μ_bar_t[2])],
            [μ_bar_t[1] + rit*np.sin(phit + μ_bar_t[2])],
            [sit]])
        
        for k in Nt_1:
            μ_bar_k = μ_bar_Nt1[k]
            delta_k = np.array([[μ_bar_k[0] - μ_bar_t[0]],
                            [μ_bar_k[1] - μ_bar_t[1]]])
            
            q_k = delta_k.T @ delta_k

            z_hat_tk = np.array([[np.sqrt(q_k)],
                                [np.arctan2(delta_k[1], delta_k[0]) - μ_bar_t[2]],
                                [μ_bar_k[2]]])
            
            Fxk = np.block([[np.eye(3), np.zeros((3, 3*k-3)), np.zeros(3, 3), np.zeros((3, 3*(N-k)))],
                            [np.zeros((3, 3)), np.zeros((3, 3*k-3)), np.eye(3), np.zeros((3, 3*(N-k)))]])
            
            Htk = (np.array([[-np.sqrt(q_k)*delta_k[0], -np.sqrt(q_k)*delta_k[1], 0, np.sqrt(q_k)*delta_k[0], np.sqrt(q_k)*delta_k[1], 0],
                            [delta_k[1], -delta_k[0], -q_k, -delta_k[1], delta_k[0], 0],
                            [0, 0, 0, 0, 0, 1]]) / q_k) @ Fxk

            Ψ_k = Htk @ Σ_bar_t @ Htk.T + Qt
            π_k = (zt[i] - z_hat_tk).T @ np.linalg.inv(Ψ_k) @ (zt[i] - z_hat_tk)
            
        π_Nt1 = alpha ######

        j_i = np.argmin(π_k)
        Nt = max(Nt, j_i)

        Kit = Σ_bar_t @ Htji.T @ np.linalg.inv(Ψ_k_ji)
        μ_bar_t = μ_bar_t + Kit @ (zt[i] - z_hat_tji)
        Σ_bar_t = (np.eye(3+3*N) - Kit @ Htji) @ Σ_bar_t
        
    μt = μ_bar_t
    Σt = Σ_bar_t
    return μt, Σt
