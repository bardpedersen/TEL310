#!/usr/bin/env python3 
import numpy as np


def EKF_SLAM_known_correspondences(μt_1, Σt_1, ut, zt, ct, number_of_landmarks, μ_bar_j, sigma):
    """
    Algorithm for calculating the EKF SLAM with known correspondences.

    μt_1 = [xt_1, yt_1, θt_1] is the mean of the robot pose at time t-1
    Σt_1 is the covariance of the robot pose at time t-1 (3x3)
    ut = [vt, wt] is the control input
    zt = {zt1,zt2,...} ranges to landmarks, where each zti = [rti, phiti, sti] is the range, bearing, and signature of the landmark
    ct = {ct1,ct2,...} is the correspondence of each landmark
    number_of_landmarks is the number of landmarks in the map
    μ_bar_j is the previus seen landmarks, where μ_bar_j = [xj, yj, sj] is the x, y and signature of the landmark pose
    sigma = [sigma_r, sigma_phi, sigma_s] is the standard deviation of the measurements    

    Returns:
    μt is the mean of the robot pose at time t
    Σt is the covariance of the robot pose at time t (3x3)
    """
    theta = μt_1[2]
    vt, wt = ut
    sigma_r, sigma_phi, sigma_s = sigma
    N = number_of_landmarks
    dt = 1  # time step
    Rt = np.diagflat(np.array([5.0, 5.0, 100.0])) ** 2

    Fx = np.block([np.eye(3), np.zeros((3, 3 * N))])

    μ_bar_t = μt_1 + np.dot(Fx.T, np.array([[-vt / wt * np.sin(theta) + vt / wt * np.sin(theta + wt * dt)],
                                           [vt / wt * np.cos(theta) - vt / wt * np.cos(theta + wt * dt)],
                                           [wt * dt]],dtype=object))  # 3+3N x 1

    μ_bar_t = np.array([arr[0].item() for arr in μ_bar_t])

    Gt = np.eye(3 + 3 * N) + np.dot(Fx.T, np.array([[0, 0, -vt / wt * np.cos(theta) + vt / wt * np.cos(theta + wt * dt)],
                                                    [0, 0, -vt / wt * np.sin(theta) + vt / wt * np.sin(theta + wt * dt)],
                                                    [0, 0, 0]], dtype=object)) @ Fx  # 3+3N x 3+3N

    Σ_bar_t = np.dot(np.dot(Gt, Σt_1), Gt.T) + np.dot(Fx.T, np.dot(Rt, Fx))

    Qt = np.array([[sigma_r**2, 0, 0],
                   [0, sigma_phi**2, 0],
                   [0, 0, sigma_s**2]], dtype=object)  # 3x3

    for i, zti in enumerate(zt):
        j = len(ct)
        rit, phit, sit = zti

        if sit not in μ_bar_j[::3]:  # Check every third element in μ_bar_j, where sit should be
            μ_bar_j[(sit + 1) * 3] = μ_bar_t[0] + rit * np.cos(phit + μ_bar_t[2])
            μ_bar_j[(sit + 1) * 3 + 1] = μ_bar_t[1] - rit * np.sin(phit + μ_bar_t[2])
            μ_bar_j[(sit + 1) * 3 + 2] = sit

        delta = np.array([sum(μ_bar_j[3::3] - μ_bar_t[3::3]),  # All x values except first for robot
                            sum(μ_bar_j[4::3] - μ_bar_t[4::3])])  # All y values except second for robot

        q = np.dot(delta.T, delta)

        z_hat_it = np.array([[np.sqrt(q)],
                                [np.arctan2(delta[1], delta[0]) - μ_bar_t[2]],
                                [μ_bar_j[2]]])

        Fxj = np.block([[np.eye(3), np.zeros((3, 3 * j - 3)), np.zeros((3, 3)), np.zeros((3, 3 * (N - j)))],
                        [np.zeros((3, 3)), np.zeros((3, 3 * j - 3)), np.eye(3), np.zeros((3, 3 * (N - j)))]])
    
        Hit = (np.dot(np.array([[-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], 0, np.sqrt(q) * delta[0],
                                    np.sqrt(q) * delta[1], 0],
                                    [delta[1], -delta[0], -q, -delta[1], delta[0], 0],
                                    [0, 0, 0, 0, 0, q]]), Fxj)) / q

        temp = np.dot(np.dot(Hit, Σ_bar_t), Hit.T) + Qt
        temp = np.array([[x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x for x in row] for row in temp])
        Kit = np.dot(np.dot(Σ_bar_t, Hit.T), np.linalg.inv(temp)) 
        μ_bar_t = np.reshape(μ_bar_t, (3 + 3 * N, 1))
        μ_bar_t = μ_bar_t + np.dot(Kit, (np.reshape([rit, phit, sit], (3, 1)) - z_hat_it))
        Σ_bar_t = np.dot((np.eye(3 + 3 * N) - np.dot(Kit, Hit)), Σ_bar_t)  # 3x3
        μ_bar_t = [x[0] for sublist in μ_bar_t for x in sublist]            
            
    return μ_bar_t, Σ_bar_t, μ_bar_j



def EKF_SLAM(μt_1, Σt_1, ut, zt, Nt_1, μ_bar_Nt1, sigma):
    """
    Algorithm for calculating the EKF SLAM with unknown correspondences.

    μt_1 = [xt_1, yt_1, θt_1] is the mean of the robot pose at time t-1
    Σt_1 is the covariance of the robot pose at time t-1 (3x3)
    ut = [vt, wt] is the control input
    zt = {zt1,zt2,...} ranges to landmarks, where each zti = [rti, phiti, sti] is the range, bearing, and signature of the landmark
    Nt_1 is the number of landmarks at time t-1
    μ_bar_Nt1 is the previus seen landmarks, where μ_bar_Nt1 = [xj, yj, sj] is the x, y and signature of the landmark pose
    sigma = [sigma_r, sigma_phi, sigma_s] is the standard deviation of the measurements

    Returns:
    μt is the mean of the robot pose at time t
    Σt is the covariance of the robot pose at time t (3x3)
    Nt is the number of landmarks at time t
    """

    theta = μt_1[2]
    vt, wt = ut
    sigma_r, sigma_phi, sigma_s = sigma
    N = Nt_1
    dt = 1
    Rt = np.diagflat(np.array([5.0, 5.0, 100.0])) ** 2

    Fx = np.block([np.eye(3), np.zeros((3, 3 * N))])

    μ_bar_t = μt_1 + np.dot(Fx.T, np.array([[-vt / wt * np.sin(theta) + vt / wt * np.sin(theta + wt * dt)],
                                           [vt / wt * np.cos(theta) - vt / wt * np.cos(theta + wt * dt)],
                                           [wt * dt]],dtype=object))  # 3+3N x 1

    μ_bar_t = np.array([arr[0].item() for arr in μ_bar_t])

    Gt = np.eye(3 + 3 * N) + np.dot(Fx.T, np.array([[0, 0, -vt / wt * np.cos(theta) + vt / wt * np.cos(theta + wt * dt)],
                                                    [0, 0, -vt / wt * np.sin(theta) + vt / wt * np.sin(theta + wt * dt)],
                                                    [0, 0, 0]], dtype=object)) @ Fx  # 3+3N x 3+3N

    Σ_bar_t = np.dot(np.dot(Gt, Σt_1), Gt.T) + np.dot(Fx.T, np.dot(Rt, Fx))

    Qt = np.array([[sigma_r**2, 0, 0],
                   [0, sigma_phi**2, 0],
                   [0, 0, sigma_s**2]], dtype=object)  # 3x3
    
    for i, (rit, phit, sit) in enumerate(zt):
        k = int((len(μ_bar_Nt1) - 3) / 3)
        
        μ_bar_Nt1[(sit + 1) * 3] = μ_bar_t[0] + rit * np.cos(phit + μ_bar_t[2])
        μ_bar_Nt1[(sit + 1) * 3 + 1] = μ_bar_t[1] - rit * np.sin(phit + μ_bar_t[2])
        μ_bar_Nt1[(sit + 1) * 3 + 2] = sit
        
        for j, value in enumerate(μ_bar_Nt1[2::3]):
            delta_k = np.array([μ_bar_Nt1[j] - μ_bar_t[0], μ_bar_Nt1[j+1] - μ_bar_t[1]])
            
            q_k = delta_k.T @ delta_k

            z_hat_tk = np.array([[np.sqrt(q_k)],
                                [np.arctan2(delta_k[1], delta_k[0]) - μ_bar_t[2]],
                                [μ_bar_Nt1[j+2]]])
            
            Fxk = np.block([[np.eye(3), np.zeros((3, 3 * k - 3)), np.zeros((3, 3)), np.zeros((3, 3 * (N - k)))],
                            [np.zeros((3, 3)), np.zeros((3, 3* k - 3)), np.eye(3), np.zeros((3, 3 * ( N - k)))]])
            
            
            Htk = (np.array([[-np.sqrt(q_k)*delta_k[0], -np.sqrt(q_k)*delta_k[1], 0, np.sqrt(q_k)*delta_k[0], np.sqrt(q_k)*delta_k[1], 0],
                            [delta_k[1], -delta_k[0], -q_k, -delta_k[1], delta_k[0], 0],
                            [0, 0, 0, 0, 0, 1]]) / q_k) @ Fxk

            Ψ_k = Htk @ Σ_bar_t @ Htk.T + Qt
            Ψ_k = np.array([[x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x for x in row] for row in Ψ_k])
        π_k = (zt[i] - z_hat_tk).T @ np.linalg.inv(Ψ_k) @ (zt[i] - z_hat_tk)

        Kit = Σ_bar_t @ Htk.T @ np.linalg.inv(Ψ_k)
        μ_bar_t = μ_bar_t.reshape((3+3*N, 1))
        zt[i] = np.array(zt[i]).reshape((3, 1))
        μ_bar_t = μ_bar_t + Kit @ (zt[i] - z_hat_tk)
        μ_bar_t = [[x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x for x in row] for row in μ_bar_t]
        μ_bar_t = np.array([x for sublist in μ_bar_t for x in sublist])
        Σ_bar_t = (np.eye(3+3*N) - Kit @ Htk) @ Σ_bar_t
        Σ_bar_t = np.array([[x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x for x in row] for row in Σ_bar_t])

    return μ_bar_t, Σ_bar_t, μ_bar_Nt1
