#!/usr/bin/env python3 

import math
import numpy as np
from scipy.linalg import sqrtm

"""
Algorithm for calculating the robot pose at time t, with a mean and a standar deviation, using the Extended Kalman Filter, with known correspondences.
μt_1 = [xt_1, yt_1, θt_1] is the mean of the robot pose at time t-1
Σt_1 is the covariance of the robot pose at time t-1

ut = [vt, wt] is the control input
zt = {zt1,zt2,...} ranges to landmarks, where each zti = [rti, phiti, sti] is the range, bearing, and signature of the landmark
ct = {ct1,ct2,...} are the landmark ids 
delta t is the time interval between t-1 and t

alpha = [a1, a2, a3, a4] are the motion noise parameters
sigma = [sigma_r, sigma_phi, sigma_s] are the measurement noise parameters

Returns the mean and covariance of the robot pose at time t, μt and Σt, and the
likelihood of the feature observation, pzt
"""
def EKF_localization_known_correspondences(μt_1, Σt_1, ut, zt, ct, delta_t, alpha, sigma):
    x, y, theta = μt_1
    vt, wt = ut
    a1, a2, a3, a4 = alpha
    sigma_r, sigma_phi, sigma_s = sigma
    wt_close_zero = 1e-4
    
    if abs(wt) > wt_close_zero:
        # Jacobian of the motion model
        Gt = np.array([[1, 0, -vt/wt*math.cos(theta) + vt/wt*math.cos(theta + wt*delta_t)],
                        [0, 1, -vt/wt*math.sin(theta) + vt/wt*math.sin(theta + wt*delta_t)],
                        [0, 0, 1]]) 

        Vt = np.array([[(-math.sin(theta) + math.sin(theta + wt*delta_t))/wt, vt*(math.sin(theta) - math.sin(theta + wt*delta_t))/wt**2 + vt*math.cos(theta + wt*delta_t)*delta_t/wt],
                        [(math.cos(theta) - math.cos(theta + wt*delta_t))/wt, -vt*(math.cos(theta) - math.cos(theta + wt*delta_t))/wt**2 + vt*math.sin(theta + wt*delta_t)*delta_t/wt],
                        [0, delta_t]]) 
        
        # Prediction
        μ_hat_t = μt_1 + np.array([-vt/wt*math.sin(theta) + vt/wt*math.sin(theta + wt*delta_t),
                            vt/wt*math.cos(theta) - vt/wt*math.cos(theta + wt*delta_t),
                            wt*delta_t]) 
        
    else: # For when wt is close to zero
        # Jacobian of the motion model
        Gt = np.array([[1, 0, -vt*math.sin(theta)*delta_t],
                        [0, 1, vt*math.cos(theta)*delta_t],
                        [0, 0, 1]]) 

        Vt = np.array([[math.cos(theta)*delta_t, -vt*math.sin(theta)*delta_t*delta_t*0.5],
                        [math.sin(theta)*delta_t, vt*math.cos(theta)*delta_t*delta_t*0.5],
                        [0, delta_t]])
        
        # Prediction
        μ_hat_t = μt_1 + np.array([vt*math.cos(theta)*delta_t,
                                    vt*math.sin(theta)*delta_t,
                                    0]) 

    # Motion noise covariance matrix from the control input
    Mt = np.array([[a1*vt**2 + a2*wt**2, 0],
                    [0, a3*vt**2 + a4*wt**2]])
    
    Σ_hat_t = Gt @ Σt_1 @ np.transpose(Gt) + Vt @ Mt @ np.transpose(Vt) 

    Qt = np.array([[sigma_r**2, 0, 0],
                    [0, sigma_phi**2, 0],
                    [0, 0, sigma_s**2]])

    pzt = 1
    for i in range(len(zt)):
        j_x, j_y, j_s = ct[i] # j_x, j_y are the coordinates of the landmark, in book represented as m[j,x]. j_s is the signature.

        q = (j_x - μ_hat_t[0])**2 + (j_y - μ_hat_t[1])**2

        z_hat_t_i = np.array([math.sqrt(q),
                            math.atan2(j_y - μ_hat_t[1], j_x - μ_hat_t[0]) - μ_hat_t[2],
                            j_s])
        
        Hti = np.array([[-(j_x - μ_hat_t[0])/math.sqrt(q), -(j_y - μ_hat_t[1])/math.sqrt(q), 0],
                        [(j_y - μ_hat_t[1])/q, -(j_x - μ_hat_t[0])/q, -1],
                        [0, 0, 0]]) # Jacobian
        
        Sti = Hti @ Σ_hat_t @ np.transpose(Hti) + Qt # the uncertainty corresponding to the predicted measurement zˆit

        Kti = Σ_hat_t @ np.transpose(Hti) @ np.linalg.inv(Sti) # Kalman gain

        μ_hat_t = μ_hat_t + Kti @ (zt[i] - z_hat_t_i)

        Σ_hat_t = (np.identity(3) - Kti @ Hti) @ Σ_hat_t

        pzt *= (np.linalg.det(2*np.pi * Sti)**(-1/2) * np.exp(-1/2 * np.transpose(zt[i] - z_hat_t_i) @ np.linalg.inv(Sti) @ (zt[i] - z_hat_t_i)))
    μt = μ_hat_t
    Σt = Σ_hat_t    
    return μt, Σt, pzt


"""
Algorithm for calculating the robot pose at time t, with a mean and a standar deviation, using the Extended Kalman Filter.
μt_1 = [xt_1, yt_1, θt_1] is the mean of the robot pose at time t-1
Σt_1 is the covariance of the robot pose at time t-1

ut = [vt, wt] is the control input
zt = {zt1,zt2,...} ranges to landmarks, where each zti = [rti, phiti, sti] is the range, bearing, and signature of the landmark

landmarks = {landmark1, landmark2,...} are the landmarks in the map, where each landmark = [x, y, signature] is the position and signature of the landmark
delta t is the time interval between t-1 and t
alpha = [a1, a2, a3, a4] are the motion noise parameters
sigma = [sigma_r, sigma_phi, sigma_s] are the measurement noise parameters

Returns the mean and covariance of the robot pose at time t, μt and Σt
"""
def EKF_localization(μt_1, Σt_1, ut, zt, landmarks, delta_t, alpha, sigma):
    theta = μt_1[2]
    vt, wt = ut
    a1, a2, a3, a4 = alpha
    sigma_r, sigma_phi, sigma_s = sigma
    wt_close_zero = 1e-4

    if abs(wt) > wt_close_zero:
        # Jacobian of the motion model
        Gt = np.array([[1, 0, -vt/wt*math.cos(theta) + vt/wt*math.cos(theta + wt*delta_t)],
                        [0, 1, -vt/wt*math.sin(theta) + vt/wt*math.sin(theta + wt*delta_t)],
                        [0, 0, 1]]) 

        Vt = np.array([[(-math.sin(theta) + math.sin(theta + wt*delta_t))/wt, vt*(math.sin(theta) - math.sin(theta + wt*delta_t))/wt**2 + vt*math.cos(theta + wt*delta_t)*delta_t/wt],
                        [(math.cos(theta) - math.cos(theta + wt*delta_t))/wt, -vt*(math.cos(theta) - math.cos(theta + wt*delta_t))/wt**2 + vt*math.sin(theta + wt*delta_t)*delta_t/wt],
                        [0, delta_t]]) 
        
        # Prediction
        μ_hat_t = μt_1 + np.array([-vt/wt*math.sin(theta) + vt/wt*math.sin(theta + wt*delta_t),
                            vt/wt*math.cos(theta) - vt/wt*math.cos(theta + wt*delta_t),
                            wt*delta_t]) 
        
    else: # For when wt is close to zero
        # Jacobian of the motion model
        Gt = np.array([[1, 0, -vt*math.sin(theta)*delta_t],
                        [0, 1, vt*math.cos(theta)*delta_t],
                        [0, 0, 1]]) 

        Vt = np.array([[math.cos(theta)*delta_t, -vt*math.sin(theta)*delta_t*delta_t*0.5],
                        [math.sin(theta)*delta_t, vt*math.cos(theta)*delta_t*delta_t*0.5],
                        [0, delta_t]])
        
        # Prediction
        μ_hat_t = μt_1 + np.array([vt*math.cos(theta)*delta_t,
                                    vt*math.sin(theta)*delta_t,
                                    0]) 
    
    Mt = np.array([[a1*vt**2 + a2*wt**2, 0],
                    [0, a3*vt**2 + a4*wt**2]])

    Σ_hat_t = Gt @ Σt_1 @ np.transpose(Gt) + Vt @ Mt @ np.transpose(Vt)

    Qt = np.array([[sigma_r**2, 0, 0],
                    [0, sigma_phi**2, 0],
                    [0, 0, sigma_s**2]])

    Htj_list = []
    Stj_list = []
    z_hat_tj_list = []
    j_list = []
    for zti in zt: # zti = [rti, phiti, sti]
        for k in landmarks: # k = [x, y, signature]

            q = (k[0] - μ_hat_t[0])**2 + (k[1] - μ_hat_t[1])**2

            z_hat_t_k = np.array([math.sqrt(q),
                                math.atan2(k[1] - μ_hat_t[1], k[0] - μ_hat_t[0]) - μ_hat_t[2],
                                k[2]])
            
            Htk = np.array([[-(k[0] - μ_hat_t[0])/math.sqrt(q), -(k[1] - μ_hat_t[1])/math.sqrt(q), 0],
                            [(k[1] - μ_hat_t[1])/q, -(k[0] - μ_hat_t[0])/q, -1],
                            [0, 0, 0]])
            
            Stk = Htk @ Σ_hat_t @ np.transpose(Htk) + Qt

            Htj_list.append(Htk)
            Stj_list.append(Stk)
            z_hat_tj_list.append(z_hat_t_k)
            j_list.append(np.linalg.det(2*np.pi * Stk)**(-1/2) * np.exp(-1/2 * np.transpose(zti - z_hat_t_k) @ np.linalg.inv(Stk) @ (zti - z_hat_t_k)))

        max_index = j_list.index(max(j_list))
        Kti = Σ_hat_t @ np.transpose(Htj_list[max_index]) @ np.linalg.inv(Stj_list[max_index])
        μ_hat_t = μ_hat_t + Kti @ (zti - z_hat_tj_list[max_index])
        Σ_hat_t = (np.identity(3) - Kti @ Htj_list[max_index]) @ Σ_hat_t

    μt = μ_hat_t
    Σt = Σ_hat_t
    return μt, Σt


"""
Algorithm for calculating the robot pose at time t, with a mean and a standar deviation, using the Unscented Kalman Filter.

"""
def UKF_localization(μt_1, Σt_1, ut, zt, alpha, sigma):
    vt, wt = ut
    a1, a2, a3, a4 = alpha
    sigma_r, sigma_phi, sigma_s = sigma

    # Generate augmented mean and covariance
    Mt = np.array([[a1*vt**2 + a2*wt**2, 0],
                    [0, a3*vt**2 + a4*wt**2]])
    
    Qt = np.array([[sigma_r**2, 0],
                    [0, sigma_phi**2]])
    
    μ_hat_ta_1 = np.transpose(np.array(μt_1).T,  ########################
                          np.array([0, 0]).T, 
                          np.array([0, 0]).T)
    
    Σ_hat_ta_1 = np.block([[Σt_1, np.zeros((Σt_1.shape[0], Mt.shape[1])), np.zeros((Σt_1.shape[0], Qt.shape[1]))],
                       [np.zeros((Mt.shape[0], Σt_1.shape[1])), Mt, np.zeros((Mt.shape[0], Qt.shape[1]))],
                       [np.zeros((Qt.shape[0], Σt_1.shape[1])), np.zeros((Qt.shape[0], Mt.shape[1])), Qt]])
    
    # Generate Weights
    alpha = 1 # Scaling parameters
    k = 1 # Scaling parameters
    beta = 2 # if the distribution is Gaussian, beta = 2 is optimal
    Wm = []
    Wc = []
    L = len(np.diagonal(Σ_hat_ta_1)) # The dimensionality L of the augmented state is given by the sum of the state, control, and measurement dimensions, which is diagnoal elements of Σ_hat_ta_1, which is 3 + 2 + 2 = 7
    lambda_ = alpha**2 * (L+k)-L
    gamma = math.sqrt(L + lambda_)
    for i in range(2*L):
        if i == 0:
            Wm.appned(lambda_/(0 + lambda_))
            Wc.append(lambda_/(0 + lambda_) + (1 - alpha**2 + beta))
        Wm.append(1 / (2*(i + lambda_)))
        Wc.append(1 / (2*(i + lambda_)))

    # Generate sigma points
    X_ta_1 = np.array([μ_hat_ta_1, μ_hat_ta_1 + gamma*sqrtm(Σ_hat_ta_1), μ_hat_ta_1 - gamma*sqrtm(Σ_hat_ta_1)])
    X_tx_1 = np.linalg.transpose(X_ta_1[0])
    X_tu = np.linalg.transpose(X_ta_1[1])
    X_tz = np.linalg.transpose(X_ta_1[2]) # From 7.28

    # Predicton of sigma points
    X_hat_tx = g(ut+X_tu, X_tx_1) # nonlinear function ########################

    # Predicted mean
    μ_hat_t = sum([Wm[i]*X_hat_tx[i] for i in range(2*L)]) # In range 2L ?

    # Predicted covariance
    Σ_hat_t = sum([Wc[i]* (X_hat_tx[i] - μ_hat_t) @ np.transpose(X_hat_tx[i] - μ_hat_t) for i in range(2*L)])

    # Measurement sigma points
    Z_hat_t = h(X_hat_tx) + X_tz ########################

    # Predicted measurement mean
    z_hat_t = sum([Wm[i]*Z_hat_t[i] for i in range(2*L)])

    # Predicted measurement covariance
    St = sum([Wc[i]*(Z_hat_t[i] - z_hat_t) @ np.transpose(Z_hat_t[i] - z_hat_t) for i in range(2*L)])

    # Cross covariance
    Σ_hat_txz = sum([Wc[i]*(X_hat_tx[i] - μ_hat_t) @ np.transpose(Z_hat_t[i] - z_hat_t) for i in range(2*L)])

    # Update mean and covariance
    Kt = Σ_hat_txz @ np.linalg.inv(St)
    μt = μ_hat_t + Kt @ (zt - z_hat_t)
    Σt = Σ_hat_t - Kt @ St @ np.transpose(Kt)
    pzt = np.linalg.det(2*np.pi * St)**(-1/2) * np.exp(-1/2 * np.transpose(zt - z_hat_t) @ np.linalg.inv(St) @ (zt - z_hat_t))

    return μt, Σt, pzt
