from time import sleep
import cv2
import pickle
import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize.nonlin import Jacobian
from scipy.signal import resample
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import scipy.optimize

def resample(X, Y):
    st = -0.25
    en = +0.25
    gap = 0.001
    T = []
    i = 0
    while st + gap * i <= en:
        T.append(st + gap * i)
        i += 1
    if len(T) == 0 or len(X) == 0 or not(X[0] <= T[0] and T[-1] <= X[-1]):
        return [], []
    Y_ = []
    j = 0
    for i in range(len(T)):
        while j+1 < len(X) and X[j+1] <= T[i]:
            j += 1
        k = (T[i] - X[j]) / (X[j+1] - X[j])
        y = Y[j] * (1-k) + Y[j+1] * k
        Y_.append(y)
    return T, Y_

def check_contact(board_contacts, i): # check if the contact is legal or not (at least last for 30 ms)
    id = -1
    for c in board_contacts[i]:
        if c[1] == 1:
            id = c[0]

    for j in range(i+1, min(i+10,len(board_contacts))):
        is_exist = False
        for c in board_contacts[j]:
            if c[0] == id:
                is_exist = True
        if not is_exist:
            return False

    return True

def check_location(load_path):
    RANGE = 0.3

    [board_timestamps, board_contacts] = pickle.load(open(load_path + 'board_data.pickle', 'rb'))
    [camera_timestamps, fingertip_locations] = pickle.load(open(load_path + 'camera_data_re.pickle', 'rb'))
    imu_data = pickle.load(open(load_path + 'imu_data.pickle', 'rb'))
    imu_timestamps = []
    imu_accers = []
    for i in range(len(imu_data)):
        imu_timestamps.append(imu_data[i][0])
        ax, ay, az = imu_data[i][4:7]
        gra_x, gra_y, gra_z = imu_data[i][7:10]
        accer = ax * gra_x + ay * gra_y + az * gra_z
        imu_accers.append(accer)

    Ts = []
    Xs = []
    As = []
    j = 0 # index for locations
    k = 0 # index for imu
    for i in range(len(board_timestamps)):
        is_touch_down = False
        for c in board_contacts[i]:
            if c[1] == 1 and check_contact(board_contacts, i): # touch down
                is_touch_down = True

        if is_touch_down:
            touch_time = board_timestamps[i]

            while (k+1 < len(imu_timestamps) and imu_timestamps[k+1] <= touch_time - RANGE):
                k += 1
            T = []
            A = []
            l = k
            while (l < len(imu_timestamps) and imu_timestamps[l] <= touch_time + RANGE):
                T.append(imu_timestamps[l] - touch_time)
                a = imu_accers[l] * 9.8 # g --> m / s^2
                A.append(a)
                l += 1
            T, A = resample(T, A)
            if len(T) != 0:
                Ts.append(T)
                As.append(A)

            while (j+1 < len(camera_timestamps) and camera_timestamps[j+1] <= touch_time - RANGE):
                j += 1
            T = []
            X = []
            l = j
            while (l < len(camera_timestamps) and camera_timestamps[l] <= touch_time + RANGE):
                if fingertip_locations[l][1] != -1:
                    T.append(camera_timestamps[l] - touch_time)
                    x = fingertip_locations[l][1] * 0.01 # cm --> m
                    #x = fingertip_locations[l][1]
                    X.append(x)
                l += 1
            T, X = resample(T, X)
            if len(T) != 0:
                X = np.array(X) - np.min(X)
                Ts.append(T)
                Xs.append(X)
    
    if len(Xs) == 0:
        return
    Ts = np.array(Ts)
    Xs = np.array(Xs)
    As = np.array(As)
    T = Ts[0,:]
    X = np.array([np.mean(Xs[:,i]) for i in range(len(T))])
    A = np.array([np.mean(As[:,i]) for i in range(len(T))])
    # plt.plot(T,X)
    # plt.plot(T,A)
    # plt.show()

    for id in range(5):
        T_ = []
        X_ = []
        A_ = []
        for i in range(len(T)):
            t = T[i]
            x = Xs[id][i]
            a = As[id][i]
            if (t >= -0.06 and t <= -0.01):
                T_.append(t)
                X_.append(x)
                A_.append(a)
        T_ = np.array(T_) - T_[0]

        x0 = 0.01
        x1 = -0.01
        a = -2.5
        ts = 0.01 # The timestamp of the first frame
        t1 = 0.2 # the duration of the whole touch
        
        guess = np.array((x0, x1, a, ts, t1))
        cons = (
            # {'type': 'ineq', 'fun': con_ineq1},
            # {'type': 'ineq', 'fun': con_ineq2},
            # {'type': 'ineq', 'fun': con_ineq3},
            # {'type': 'ineq', 'fun': con_ineq4}
        )
        bnds = ((0,0.03), (-0.03,0), (-5,0), (0.00,0.03), (0.05,0.3))
        res = minimize(objective, guess, args=(T_, X_, A_), method='powell', constraints=cons, bounds=bnds)

        print(res.fun)
        print(res.success)
        args = res.x
        x0, x1, a, ts, t1 = args
        print('x0=%.3f, x1=%.3f, a=%.3f, ts=%.3f, t1=%.3f' % (x0, x1, a, ts, t1))

        T_pred = []
        X_pred = []
        A_pred = []
        for i in range(300):
            t = i * 0.001
            if t > t1:
                break
            r = t / t1
            x = x0 + (x1 - x0) * S_r(r, args)
            a = (x1 - x0) * A_r(r, args) / (t1**2)
            T_pred.append(t)
            X_pred.append(x)
            A_pred.append(a)
        plt.subplot(1, 2, 1)
        plt.plot(T_pred, X_pred)
        plt.plot(T_+ts, X_)
        plt.subplot(1, 2, 2)
        plt.plot(T_pred, A_pred)
        plt.plot(T_+ts, A_)

        plt.show()

def S_r(r, args):
    x0, x1, a, ts, t1 = args
    a3 = 10 + a * 0.5
    a4 = -15 - a
    a5 = 6 + a * 0.5
    x = a3 * r**3 + a4 * r**4 + a5 * r**5
    return x

def V_r(r, args):
    x0, x1, a, ts, t1 = args
    a3 = 10 + a * 0.5
    a4 = -15 - a
    a5 = 6 + a * 0.5
    v = 3 * a3 * r**2 + 4 * a4 * r**3 + 5 * a5 * r**4
    return v

def A_r(r, args):
    x0, x1, a, ts, t1 = args
    a3 = 10 + a * 0.5
    a4 = -15 - a
    a5 = 6 + a * 0.5
    a = 6 * a3 * r + 12 * a4 * r**2 + 20 * a5 * r**3
    return a

def objective(args, T, X, A):
    x0, x1, a, ts, t1 = args
    a3 = 10 + a * 0.5
    a4 = -15 - a
    a5 = 6 + a * 0.5
    N = len(T)
    error = 0
    k = 5e-8 # Acc
    for i in range(N):
        r = (ts + T[i]) / t1
        x = x0 + (x1 - x0) * S_r(r, args)
        error += (X[i] - x) ** 2
    for i in range(N):
        r = (ts + T[i]) / t1
        a = (x1 - x0) * A_r(r, args) / (t1**2)
        error += (A[i] - a) ** 2 * k
    return error

# def con_ineq1(args):
#     a0, a3, a4, a5, t1, t_st = args[:6]
#     return A_t(t1, args)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('[Usage] python check.py userName-taskId')
        exit()
    load_path = 'data/' + sys.argv[1] + '/'
    check_location(load_path)
