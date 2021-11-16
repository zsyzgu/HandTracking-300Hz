from time import sleep
import cv2
import pickle
import time

def main(load_path):
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

    board_timestamps = pickle.load(open(load_path + 'board_timestamps.pickle', 'rb'))
    board_contacts = pickle.load(open(load_path + 'board_contacts.pickle', 'rb'))
    camera_timestamps = pickle.load(open(load_path + 'camera_timestamps.pickle', 'rb'))

    # ===== Illustration =====

    speed = 0.1
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
        cv2.waitKey(int(1000.0 / FPS))
        t += (1.0 / FPS) * speed

if __name__ == '__main__':
    main('data/')
    
