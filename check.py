import cv2
import pickle

def input(load_path):
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

    print(len(board_frames), len(board_timestamps), len(board_contacts), len(camera_frames), len(camera_timestamps))

if __name__ == '__main__':
    input('data/')
