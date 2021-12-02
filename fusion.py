from time import sleep
import cv2
import pickle
import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample

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

def check_location(load_path):
    RANGE = 0.3

    [board_timestamps, board_contacts] = pickle.load(open(load_path + 'board_data.pickle', 'rb'))
    [camera_timestamps, fingertip_locations] = pickle.load(open(load_path + 'camera_data.pickle', 'rb'))
    imu_data = pickle.load(open(load_path + 'imu_data.pickle', 'rb'))
    imu_timestamps = []
    imu_accers = []
    for i in range(len(imu_data)):
        imu_timestamps.append(imu_data[i][0])
        ax, ay, az = imu_data[i][4:7]
        gra_x, gra_y, gra_z = imu_data[i][7:10]
        accer = ax * gra_x + ay * gra_y + az * gra_z
        imu_accers.append(accer)

    Xs = []
    Ys = []
    j = 0 # index for locations
    k = 0 # index for imu
    for i in range(len(board_timestamps)):
        is_touch_down = False
        for c in board_contacts[i]:
            if c[1] == 1: # touch down
                is_touch_down = True

        if is_touch_down:
            touch_time = board_timestamps[i]

            while (k+1 < len(imu_timestamps) and imu_timestamps[k+1] <= touch_time):
                k += 1
            for t in range(k-15,k+15):
                if t >= 0 and t+1 < len(imu_accers) and imu_accers[t] < 0 and imu_accers[t+1] >= 0:
                    #touch_time = imu_timestamps[t]
                    break

            while (j+1 < len(camera_timestamps) and camera_timestamps[j+1] <= touch_time - RANGE):
                j += 1
            X = []
            Y = []
            l = j
            while (l < len(camera_timestamps) and camera_timestamps[l] <= touch_time + RANGE):
                X.append(camera_timestamps[l] - touch_time)
                Y.append(fingertip_locations[l][1])
                l += 1
            X, Y = resample(X, Y)
            if len(X) != 0:
                Y = np.array(Y) - np.min(Y)
                Xs.append(X)
                Ys.append(Y)
    
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    X = Xs[0,:]
    Y = np.array([np.mean(Ys[:,i]) for i in range(len(X))])
    S = np.array([np.std(Ys[:,i]) for i in range(len(X))])
    print(assess(X,Ys))
    # for i in range(len(Ys)):
    #     plt.plot(X,Ys[i])
    plt.plot(X,Y)
    plt.plot(X,Y+S)
    plt.plot(X,Y-S)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('[Usage] python check.py userName-taskId')
        exit()
    load_path = 'data/' + sys.argv[1] + '/'
    #check_imu(load_path)
    check_location(load_path)
