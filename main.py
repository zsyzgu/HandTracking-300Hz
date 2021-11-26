import multiprocessing
import pickle
import time
import sys
import os
from camera import Camera
from board import Board
from imu import IMU

def process_imu(save_path):
    imu = IMU(save_path)
    imu.run()

def process_camera(save_path):
    camera = Camera(save_path)
    camera.run()

def process_board(save_path):
    board = Board(save_path)
    board.run()

def sync_timestamps(file_path, gap):
    timestamps = pickle.load(open(file_path, 'rb'))
    timestamps = [t + gap for t in timestamps]
    pickle.dump(timestamps, open(file_path, 'wb'))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('[Usage] python main.py userName-taskId')
        exit()
        # taskId
        # (1) single tap * 50
        # (2) continuous tap * 50
        # (3) long press * 50
        # (4) drag * 50
        # (5) slide * 50
        # (6) in-air tap * 50
    save_path = 'data/' + sys.argv[1] + '/'
    os.mkdir(save_path)
    p_imu = multiprocessing.Process(target=process_imu, args=(save_path,))
    p_camera = multiprocessing.Process(target=process_camera, args=(save_path,))
    p_board = multiprocessing.Process(target=process_board, args=(save_path,))
    p_imu.start()
    start_time_imu = time.perf_counter()
    p_camera.start()
    start_time_camera = time.perf_counter()
    p_board.start()
    start_time_board = time.perf_counter()
    p_imu.join()
    p_camera.join()
    p_board.join()
    sync_timestamps(save_path + 'camera_timestamps.pickle', start_time_camera - start_time_imu)
    sync_timestamps(save_path + 'board_timestamps.pickle', start_time_board - start_time_imu)
