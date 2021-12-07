from time import sleep
import cv2
import pickle
import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from tracker import Tracker

def re_track(load_path):
    [camera_timestamps, fingertip_locations] = pickle.load(open(load_path + 'camera_data.pickle', 'rb'))

    camera_cap = cv2.VideoCapture(load_path + 'camera.avi')
    camera_frames = []
    while True:
        ret, frame = camera_cap.read()
        if frame is None:
            break
        camera_frames.append(frame)

    tracker = Tracker()
    new_fingertip_locations = []
    for i in range(len(camera_frames)):
        frame = camera_frames[i][:,:,0]
        H, W = np.shape(frame)
        frame0 = frame[:H//2,:]
        frame1 = frame[H//2:,:]
        tracker.update(frame0, frame1)
        new_fingertip_locations.append(tracker.get_location())
        if i % 20 == 0:
            cv2.imshow('illuL', tracker.illuL)
            cv2.imshow('illuR', tracker.illuR)
            cv2.waitKey(1)

    pickle.dump([camera_timestamps, new_fingertip_locations], open(load_path + 'camera_data_re.pickle', 'wb'))

    evaluate(fingertip_locations)
    evaluate(new_fingertip_locations)

def evaluate(fingertip_locations):
    fingertip_locations = np.array(fingertip_locations)
    cnt = np.sum(fingertip_locations[:,0]==-1)
    print(cnt, len(fingertip_locations))
    #X = fingertip_locations[:,0]
    #Y = fingertip_locations[:,1]
    #plt.plot(X, Y)
    #plt.show()

if __name__ == '__main__':
    # dirs = os.listdir('./data')
    # for dir in dirs:
    #     print(dir)
    #     load_path = 'data/' + dir + '/'
    #     re_track(load_path)
    # exit()

    if len(sys.argv) != 2:
        print('[Usage] python check.py userName-taskId')
        exit()
    load_path = 'data/' + sys.argv[1] + '/'
    re_track(load_path)
