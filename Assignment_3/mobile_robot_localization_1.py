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


def g(ut, X_tu, X_tx_1, L):
    X_t = []
    for i in range(2*L+1):
        vit = ut[0] + X_tu[0][i]
        wit = ut[1] + X_tu[1][i]
        thetait = X_tx_1[2][i]
        X_t.append([X_tx_1[0][i] - vit/wit*math.sin(thetait) + vit/wit*math.sin(thetait + wit),
                   X_tx_1[1][i] + vit/wit*math.cos(thetait) - vit/wit*math.cos(thetait + wit),
                   X_tx_1[2][i] + wit])
    return np.array(X_t)

def h(Xt, X_tz, L, zt):
    mx = zt[0]
    my = zt[1]
    Zt = []
    for i in range(2*L+1):
        Zt.append([math.sqrt((mx - Xt[i][0])**2 + (my - Xt[i][1])**2) + X_tz[0][i], 
                   math.atan2(my - Xt[i][1], mx - Xt[i][0]) - Xt[i][2] + X_tz[1][i]])

    return np.array(Zt)


"""
Algorithm for calculating the robot pose at time t, with a mean and a standar deviation, using the Unscented Kalman Filter.

"""
def UKF_localization(μt_1, Σt_1, ut, zt, alpha, sigma):
    vt, wt = ut
    zt = np.array(zt)
    a1, a2, a3, a4 = alpha
    sigma_r, sigma_phi, sigma_s = sigma

    # Generate augmented mean and covariance
    Mt = np.array([[a1*vt**2 + a2*wt**2, 0],
                    [0, a3*vt**2 + a4*wt**2]])
    
    Qt = np.array([[sigma_r**2, 0],
                    [0, sigma_phi**2]])
    
    μ_ta_1 = np.array([[μt_1[0]],[μt_1[1]],[μt_1[2]],
                           [0], [0],
                           [0], [0]])
    
    Σ_ta_1 = np.block([[abs(Σt_1), np.zeros((Σt_1.shape[0], Mt.shape[1])), np.zeros((Σt_1.shape[0], Qt.shape[1]))],
                       [np.zeros((Mt.shape[0], Σt_1.shape[1])), abs(Mt), np.zeros((Mt.shape[0], Qt.shape[1]))],
                       [np.zeros((Qt.shape[0], Σt_1.shape[1])), np.zeros((Qt.shape[0], Mt.shape[1])), abs(Qt)]])
    
    # Generate Weights
    alpha = 1 # Scaling parameters
    k = 0 # Scaling parameters
    beta = 2 # if the distribution is Gaussian, beta = 2 is optimal
    L = len(np.diagonal(Σ_ta_1)) # The dimensionality L of the augmented state is given by the sum of the state, control, and measurement dimensions, which is diagnoal elements of Σ_hat_ta_1, which is 3 + 2 + 2 = 7
    lambda_ = alpha**2 * (L+k) - L 
    Wm = np.zeros(2*L + 1)
    Wc = np.zeros(2*L + 1)
    gamma = math.sqrt(L + lambda_)
    Wm[:] = .5/(L + lambda_)
    Wc[:] = .5/(L + lambda_)
    Wm[0] = lambda_/(lambda_ + L)
    Wc[0] = lambda_/(lambda_ + L) + (1 - alpha**2 + beta)

    # Generate sigma points
    temp = np.linalg.cholesky(Σ_ta_1).T
    X_ta_1 = np.column_stack([μ_ta_1, (μ_ta_1 + gamma*temp.T).T, (μ_ta_1 - gamma*temp.T).T])
    X_tx_1 = X_ta_1[0:3] # x, y, θ
    X_tu = X_ta_1[3:5] # vt, wt noise component
    X_tz = X_ta_1[-2:] # r, phi #From 7.28

    # Predicton of sigma points
    X_bar_tx = g(ut, X_tu, X_tx_1, L) #ut+X_tu, X_tx_1  # 3x15
    print("X_bar_tx", X_bar_tx)

    # Predicted mean
    μ_bar_t = 0 
    for i in range(2*L+1):
        μ_bar_t += Wm[i]*X_bar_tx[i]
    μ_bar_t = np.array(μ_bar_t) # x, y, θ, 3x1

    # Predicted covariance
    Σ_bar_t = 0
    for i in range(2*L+1):
        matrix = np.array([[X_bar_tx[i][0] - μ_bar_t[1]], [X_bar_tx[i][1] - μ_bar_t[1]], [X_bar_tx[i][2] - μ_bar_t[2]]])
        Σ_bar_t += (Wc[i]* (np.array([X_bar_tx[i] - μ_bar_t]).T @ matrix.T))
    Σ_bar_t = np.array(Σ_bar_t) # x, y, θ, 3x3

    # Measurement sigma points
    Z_bar_t = h(X_bar_tx, X_tz, L, zt) # 2x15

    # Predicted measurement mean
    z_hat_t = 0
    for i in range(2*L+1):
        z_hat_t += (Wm[i]*Z_bar_t[i])
    z_hat_t = np.array([[z_hat_t[0]], [z_hat_t[1]]]) # 2x1

    St = 0
    for i in range(2*L+1):
        St += Wc[i]*(Z_bar_t[i] - z_hat_t) @ np.transpose(Z_bar_t[i] - z_hat_t)
    St = np.array(St) # 2x2 

    # Cross covariance, not matrix
    Σ_hat_txz = 0
    for i in range(2*L+1):
        matrix1 = np.array([Z_bar_t[i][0] - z_hat_t[0], Z_bar_t[i][1] - z_hat_t[1]])
        matrix2 = np.array([X_bar_tx[i][0] - μ_bar_t[0], X_bar_tx[i][1] - μ_bar_t[1], X_bar_tx[i][2] - μ_bar_t[2]])
        matrix2 = matrix2.reshape((3, 1))
        Σ_hat_txz += Wc[i]* matrix2 @  np.transpose(matrix1) 
    Σ_hat_txz = np.array(Σ_hat_txz) # 3x2

    # Update mean and covariance
    Kt = Σ_hat_txz @ St
    μt = μ_bar_t.reshape(3, 1) + Kt @ (zt.reshape((2, 1)) - z_hat_t.reshape((2, 1)))
    Σt = Σ_bar_t - Kt @ St @ np.transpose(Kt)
    #pzt = np.linalg.det(2*np.pi * St)**(-1/2) * np.exp(-1/2 * np.transpose(zt - z_hat_t) @ np.linalg.inv(St) @ (zt - z_hat_t))
    pzt = 1
    return μt.reshape(3,), Σt , pzt
