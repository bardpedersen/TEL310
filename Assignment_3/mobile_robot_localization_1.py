#!/usr/bin/env python3 

import math
import numpy as np

"""
Gaussian estimate of the robot pose at time t-1, with mean μt-1 and covariance Σt-1
control = ut
map = m



likelihood of the feature observation, pzt
"""
def EKF_localization_known_correspondences(μt_1, Σt_1, ut, zt, ct, m):
    theta = μt_1[2]
    vt, wt = ut
    delta_t = None ##########
    a1, a2, a3, a4 = None ##########
    sigma_r, sigma_phi, sigma_s = None ##########


    Gt = np.array([[1, 0, -vt/wt*math.cos(theta) + vt/wt*math.cos(theta + wt*delta_t)],
                    [0, 1, -vt/wt*math.sin(theta) + vt/wt*math.sin(theta + wt*delta_t)],
                    [0, 0, 1]])

    Vt = np.array([[(-math.sin(theta) + math.sin(theta + wt*delta_t))/wt, vt*(math.sin(theta) - math.sin(theta + wt*delta_t))/wt**2 + vt*math.cos(theta + wt*delta_t)*delta_t/wt],
                    [(math.cos(theta) - math.cos(theta + wt*delta_t))/wt, -vt*(math.cos(theta) - math.cos(theta + wt*delta_t))/wt**2 + vt*math.sin(theta + wt*delta_t)*delta_t/wt],
                    [0, delta_t]])
    
    Mt = np.array([[a1*vt**2 + a2*wt**2, 0],
                    [0, a3*vt**2 + a4*wt**2]])
    
    μ_hat_t = μt_1 + np.array([[-vt/wt*math.sin(theta) + vt/wt*math.sin(theta + wt*delta_t)],
                        [vt/wt*math.cos(theta) - vt/wt*math.cos(theta + wt*delta_t)],
                        [wt*delta_t]])

    Σ_hat_t = Gt @ Σt_1 @ np.transpose(Gt) + Vt @ Mt @ np.transpose(Vt)

    Qt = np.array([[sigma_r**2, 0, 0],
                    [0, sigma_phi**2, 0],
                    [0, 0, sigma_s**2]])

    for zti in zt:
        rti, phiti, sti = zti
        j = cti
        j_x, j_y, j_s = None ##########

        q = (m[j_x] - μ_hat_t[0])**2 + (m[j_y] - μ_hat_t[1])**2
        z_hat_t_i = np.array([[math.sqrt(q)],
                            [math.atan2(m[j_y] - μ_hat_t[1], m[j_x] - μ_hat_t[0]) - μ_hat_t[2]],
                            [m[j_s]]])
        
        Hti = np.array([[-(m[j_x] - μ_hat_t[0])/math.sqrt(q), -(m[j_y] - μ_hat_t[1])/math.sqrt(q), 0],
                        [(m[j_y] - μ_hat_t[1])/q, -(m[j_x] - μ_hat_t[0])/q, -1],
                        [0, 0, 0]])
        
        Sti = Hti @ Σ_hat_t @ np.transpose(Hti) + Qt

        Kti = Σ_hat_t @ np.transpose(Hti) @ np.linalg.inv(Sti)

        μ_hat_t = μ_hat_t + Kti @ (zti - z_hat_t_i)

        Σ_hat_t = (np.identity(3) - Kti @ Hti) @ Σ_hat_t

    μt = μ_hat_t
    Σt = Σ_hat_t
    
    pzt = np.prod(np.linalg.det(2*np.pi * Sti)**(-1/2) * np.exp(-1/2 * np.transpose(zti - z_hat_t_i) @ np.linalg.inv(Sti) @ (zti - z_hat_t_i))) #Need to itterate over all i
    return μt, Σt, pzt



def EKF_localization(μt_1, Σt_1, ut, zt, m):
    theta = μt_1[2]
    vt, wt = ut
    delta_t = None ##########
    a1, a2, a3, a4 = None ##########
    sigma_r, sigma_phi, sigma_s = None ##########
    landmarks = None ##########


    Gt = np.array([[1, 0, -vt/wt*math.cos(theta) + vt/wt*math.cos(theta + wt*delta_t)],
                    [0, 1, -vt/wt*math.sin(theta) + vt/wt*math.sin(theta + wt*delta_t)],
                    [0, 0, 1]])

    Vt = np.array([[(-math.sin(theta) + math.sin(theta + wt*delta_t))/wt, vt*(math.sin(theta) - math.sin(theta + wt*delta_t))/wt**2 + vt*math.cos(theta + wt*delta_t)*delta_t/wt],
                    [(math.cos(theta) - math.cos(theta + wt*delta_t))/wt, -vt*(math.cos(theta) - math.cos(theta + wt*delta_t))/wt**2 + vt*math.sin(theta + wt*delta_t)*delta_t/wt],
                    [0, delta_t]])
    
    Mt = np.array([[a1*vt**2 + a2*wt**2, 0],
                    [0, a3*vt**2 + a4*wt**2]])
    
    μ_hat_t = μt_1 + np.array([[-vt/wt*math.sin(theta) + vt/wt*math.sin(theta + wt*delta_t)],
                        [vt/wt*math.cos(theta) - vt/wt*math.cos(theta + wt*delta_t)],
                        [wt*delta_t]])

    Σ_hat_t = Gt @ Σt_1 @ np.transpose(Gt) + Vt @ Mt @ np.transpose(Vt)

    Qt = np.array([[sigma_r**2, 0, 0],
                    [0, sigma_phi**2, 0],
                    [0, 0, sigma_s**2]])

    j = [] ##########
    Htj = [] ##########
    Stj = [] ##########
    for zti in zt:
        for k in landmarks:

            q = (m[k[0]] - μ_hat_t[0])**2 + (m[k[1]] - μ_hat_t[1])**2

            z_hat_t_k = np.array([[math.sqrt(q)],
                                [math.atan2(m[k[1]] - μ_hat_t[1], m[k[0]] - μ_hat_t[0]) - μ_hat_t[2]],
                                [m[k[2]]]])
            
            Htk = np.array([[-(m[k[0]] - μ_hat_t[0])/math.sqrt(q), -(m[k[1]] - μ_hat_t[1])/math.sqrt(q), 0],
                            [(m[k[1]] - μ_hat_t[1])/q, -(m[k[0]] - μ_hat_t[0])/q, -1],
                            [0, 0, 0]])
            
            Stk = Htk @ Σ_hat_t @ np.transpose(Htk) + Qt

            Htj.append(Htk)
            Stj.append(Stk)
        j.append(max(np.linalg.det(2*np.pi * Stk)**(-1/2) * np.exp(-1/2 * np.transpose(zti - z_hat_t_k) @ np.linalg.inv(Stk) @ (zti - z_hat_t_k)))) #Need to itterate over all i
        Kti = Σ_hat_t @ np.transpose(Htj) @ np.linalg.inv(Stj) # List?
        μ_hat_t = μ_hat_t + Kti @ (zti - z_hat_t_k)

        Σ_hat_t = (np.identity(3) - Kti @ Htj) @ Σ_hat_t

    μt = μ_hat_t
    Σt = Σ_hat_t
    return μt, Σt


def UKF_localization(μt_1, Σt_1, ut, zt, m):
    vt, wt = ut
    delta_t = None ##########
    a1, a2, a3, a4 = None ##########
    sigma_r, sigma_phi, sigma_s = None ##########

    # Generate augmented mean and covariance
    Mt = np.array([[a1*vt**2 + a2*wt**2, 0],
                    [0, a3*vt**2 + a4*wt**2]])
    
    Qt = np.array([[sigma_r**2, 0],
                    [0, sigma_phi**2]])
    
    μ_hat_ta_1 = np.transpose(np.array(np.transpose(μt_1), 
                          np.transpose([0, 0]), 
                          np.transpose([0, 0])))
    
    Σ_hat_ta_1 = np.array([[Σt_1, 0, 0],
                            [0, Mt, 0],
                            [0, 0, Qt]])
    
    gamma = None ##########

    # Generate sigma points
    X_ta_1 = np.array([μ_hat_ta_1, μ_hat_ta_1 + gamma*np.sqrt(Σ_hat_ta_1), μ_hat_ta_1 - gamma*np.sqrt(Σ_hat_ta_1)])

    # Propagate sigma points through motion model and compute gaussian statistics
    X_hat_tx = g(ut+X_tu, X_tx_1)

    μ_hat_t = sum([Wm[i]*X_hat_tx[i] for i in range(2*3+1)]) # In range 2L ?

    Σ_hat_t = sum([Wc[i]*(X_hat_tx[i] - μ_hat_t) @ np.transpose(X_hat_tx[i] - μ_hat_t) for i in range(2*3+1)]) # In range 2L ?

    # Predicted observation at sigma point and compute gaussian statistics
    Z_hat_t = h(X_hat_tx) + X_tz

    z_hat_t = sum([Wm[i]*Z_hat_t[i] for i in range(2*3+1)]) # In range 2L ?

    St = sum([Wc[i]*(Z_hat_t[i] - z_hat_t) @ np.transpose(Z_hat_t[i] - z_hat_t) for i in range(2*3+1)]) # In range 2L ?

    Σ_hat_txz = sum([Wc[i]*(X_hat_tx[i] - μ_hat_t) @ np.transpose(Z_hat_t[i] - z_hat_t) for i in range(2*3+1)]) # In range 2L ?

    # Update mean and covariance
    Kt = Σ_hat_txz @ np.linalg.inv(St)

    μt = μ_hat_t + Kt @ (zt - z_hat_t)

    Σt = Σ_hat_t - Kt @ St @ np.transpose(Kt)

    pzt = np.linalg.det(2*np.pi * St)**(-1/2) * np.exp(-1/2 * np.transpose(zt - z_hat_t) @ np.linalg.inv(St) @ (zt - z_hat_t))

    return μt, Σt, pzt
