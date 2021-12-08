from time import sleep
import cv2
import pickle
import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from scipy.optimize import minimize

def prepare_imu(load_path):
    [board_timestamps, board_contacts] = pickle.load(open(load_path + 'board_data.pickle', 'rb'))
    #[camera_timestamps, fingertip_locations] = pickle.load(open(load_path + 'camera_data.pickle', 'rb'))
    imu_data = pickle.load(open(load_path + 'imu_data.pickle', 'rb'))

    Xs = []
    Ys = []
    for i in range(len(board_timestamps)):
        print(float(i)/len(board_timestamps))
        is_touch_down = False
        for c in board_contacts[i]:
            if c[1] == 1: # touch down
                is_touch_down = True
        
        if is_touch_down:
            touch_time = board_timestamps[i]
            X = []
            Y = []
            for j in range(len(imu_data)):
                t = imu_data[j][0]
                if touch_time - 0.3 <= t and t <= touch_time + 0.3:
                    ax, ay, az = imu_data[j][4], imu_data[j][5], imu_data[j][6]
                    gra_x, gra_y, gra_z = imu_data[j][7], imu_data[j][8], imu_data[j][9]
                    y = ax * gra_x + ay * gra_y + az * gra_z
                    X.append(t - touch_time)
                    Y.append(y)

            X, Y = resample(X, Y)

            if len(X) != 0:
                Xs.append(X)
                Ys.append(Y)

    X = np.array(Xs[0])
    Ys = np.array(Ys)

    pickle.dump([X,Ys], open('tmp.pickle', 'wb'))

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

def assess(X,Ys):
    Y = np.array([np.mean(Ys[:,i]) for i in range(len(X))])
    N = int(len(X) * 0.4)
    errors = []
    for i in range(len(Ys)):
        diff = Ys[i] - Y
        error = np.sum(diff[N:-N]*diff[N:-N])
        errors.append(error)
    return np.mean(errors)


def check_imu(load_path):
    prepare_imu(load_path)
    [X,Ys] = pickle.load(open('tmp.pickle', 'rb'))
    print(assess(X,Ys))

    for i in range(len(Ys)):
        shift = 0
        for t in range(-10,10):
            if Ys[i,len(X)//2+t] < 0 and Ys[i,len(X)//2+t+1] >= 0:
                shift = t

        if shift > 0:
            Ys[i,:] = np.concatenate([Ys[i,shift:], np.zeros(shift)])
        if shift < 0:
            Ys[i,:] = np.concatenate([np.zeros(-shift), Ys[i,:shift]])

    Y = np.array([np.mean(Ys[:,i]) for i in range(len(X))])

    print(assess(X,Ys))

    plt.plot(X, Y)
    plt.show()

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
                A.append(imu_accers[l])
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
                    X.append(fingertip_locations[l][1])
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
    #print(len(Xs), assess(T,Xs))
    # for i in range(len(Xs)):
    #     plt.plot(T,Xs[i])
    plt.plot(T,X)
    plt.plot(T,A)
    plt.show()

    for id in range(10):
        T_ = []
        X_ = []
        A_ = []
        for i in range(len(T)):
            t = T[i]
            x = Xs[id][i]
            a = As[id][i]
            if (t >= -0.05 and t <= -0.01):
                T_.append(t)
                X_.append(x)
                A_.append(a)

        #calc(X_, Y_)
        x0 = np.array((0.3, 0.1, 0, 0))
        res = minimize(func_x(T_, X_), x0, method='SLSQP', constraints=con(T_, X_))
        print(res.fun)
        print(res.success)
        print(res.x)
        
        plt.plot(T_,X_)
        plt.show()

def func_x(T, X):
    n = len(T)
    k = 1.0 / n
    fun = lambda x: np.sum([(X[i] - ((-15 * (x[0]*i*k+x[1])**4 + 6 * (x[0]*i*k+x[1])**5 + 10 * (x[0]*i*k+x[1])**3) * x[2] + x[3])) ** 2 for i in range(n)])
    return fun

def func_a(T, A):
    n = len(T)
    k = 1.0 / n
    fun = lambda x: np.sum([(A[i] - ((-180 * (x[0]*i*k+x[1])**2 + 120 * (x[0]*i*k+x[1])**3 + 60 * (x[0]*i*k+x[1])**1) * x[2])) ** 2 for i in range(n)])
    return fun

def con(T, X):
    cons = (
        {'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: x[1]},
        {'type': 'ineq', 'fun': lambda x: 0.5 - x[0] - x[1]}
    )
    return cons

def calc(T, X):
    kt = symbols('kt')
    bt = symbols('bt')
    kx = symbols('kx')
    bx = symbols('bx')
    a1 = symbols('a1')
    a2 = symbols('a2')
    a3 = symbols('a3')

    L = a1 * (-kt) + a2 * (-bt) + a3 * (kt + bt - 0.5)

    n = len(T)
    for i in range(n):
        t = kt * i + bt
        x = (-15 * t**4 + 6 * t**5 + 10 * t**3) * kx + bx
        L = L + (X[i] - x) ** 2
    
    dify_kt = diff(L, kt)
    dify_bt = diff(L, bt)
    dify_kx = diff(L, kx)
    dify_bx = diff(L, bx)
    dual_a1 = a1 * (-kt)
    dual_a2 = a2 * (-bt)
    dual_a3 = kt + bt - 0.5
    print(1)
    aa = solve([dify_kt, dify_bt, dify_kx, dify_bx, dual_a1, dual_a2, dual_a3], [kt, bt, kx, bx, a1, a2, a3])
    print(2)
    for i in aa:
        print(i)

if __name__ == '__main__':
    # dirs = os.listdir('./data')
    # for dir in dirs:
    #     print(dir)
    #     load_path = 'data/' + dir + '/'
    #     check_location(load_path)
    # exit()

    if len(sys.argv) != 2:
        print('[Usage] python check.py userName-taskId')
        exit()
    load_path = 'data/' + sys.argv[1] + '/'
    check_location(load_path)
