from time import sleep
import cv2
import pickle
import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def check(load_path):
    # ===== Input =====

    board_cap = cv2.VideoCapture(load_path + 'board.avi')
    board_frames = []
    while True:
        ret, frame = board_cap.read()
        if frame is None:
            break
        board_frames.append(frame)

    camera_cap = cv2.VideoCapture(load_path + 'camera.avi')
    camera_frames = []
    while True:
        ret, frame = camera_cap.read()
        if frame is None:
            break
        camera_frames.append(frame)

    [board_timestamps, board_contacts] = pickle.load(open(load_path + 'board_data.pickle', 'rb'))
    [camera_timestamps, fingertip_locations] = pickle.load(open(load_path + 'camera_data.pickle', 'rb'))
    imu_data = pickle.load(open(load_path + 'imu_data.pickle', 'rb'))

    # ===== Illustration Fingertip location =====

    for i in range(len(board_timestamps)):
        is_touch_down = False
        for c in board_contacts[i]:
            if c[1] == 1: # touch down
                is_touch_down = True
        
        if is_touch_down:
            touch_time = board_timestamps[i]
            X = []
            Y = []
            A = -1e9
            y_fail = False
            for j in range(len(camera_timestamps)):
                t = camera_timestamps[j]
                y = fingertip_locations[j][1]
                if touch_time - 0.3 <= t and t <= touch_time + 0.3:
                    if y == -1:
                        y_fail = True
                    else:
                        X.append(t - touch_time)
                        Y.append(y)
                        if touch_time - 0.05 <= t and t < touch_time:
                            A = max(A, y)

            if not y_fail:
                X = np.array(X)
                Y = np.array(Y)
                A -= np.min(Y)
                Y -= np.min(Y)
                n = len(Y)
                Y = Y * (1.0/A)

                X = X[1:]
                Y = Y[1:] - Y[:-1]

                plt.plot(X, Y)
    plt.show()

    # ===== Illustration of IMU data =====

    # for i in range(len(board_timestamps)):
    #     is_touch_down = False
    #     for c in board_contacts[i]:
    #         if c[1] == 1: # touch down
    #             is_touch_down = True
        
    #     if is_touch_down:
    #         touch_time = board_timestamps[i]
    #         X = []
    #         Y = []
    #         for j in range(len(imu_data)):
    #             t = imu_data[j][0]
    #             y = imu_data[j][5]
    #             if touch_time - 0.2 <= t and t <= touch_time + 0.2:
    #                 X.append(t - touch_time)
    #                 Y.append(y)

    #         X = np.array(X)
    #         Y = np.array(Y)
    #         plt.plot(X, Y)
    # plt.show()

    # ===== Illustration by video =====

    '''
    speed = 2.0
    FPS = 100
    t = 0
    board_cnt = 0
    camera_cnt = 0
    while t <= board_timestamps[-1] and t <= camera_timestamps[-1]:
        while board_cnt + 1 < len(board_timestamps) and board_timestamps[board_cnt + 1] < t:
            board_cnt += 1
        while camera_cnt + 1 < len(camera_timestamps) and camera_timestamps[camera_cnt + 1] < t:
            camera_cnt += 1
        cv2.imshow('board', board_frames[board_cnt])
        cv2.imshow('camera', camera_frames[camera_cnt])
        key = cv2.waitKey(int(1000.0 / FPS))
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            t = 1e9
        t += (1.0 / FPS) * speed
    '''

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('[Usage] python check.py userName-taskId')
        exit()
    save_path = 'data/' + sys.argv[1] + '/'
    check(save_path)
