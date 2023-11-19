#!/usr/bin/env python3 
import numpy as np


def EKF_SLAM_known_correspondences(μt_1, Σt_1, ut, zt, ct, sigma):
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
        μ_bar_t = μ_bar_t + Kit @ (zt[i] - z_hat_it)
        Σ_bar_t = (np.eye(3+3*N) - Kit @ Hit) @ Σ_bar_t

    μt = μ_bar_t
    Σt = Σ_bar_t
    return μt, Σt


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
