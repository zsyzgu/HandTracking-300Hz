import multiprocessing
import pickle
import time
from camera import Camera
from board import Board

def process_camera(save_path):
    camera = Camera(save_path)
    camera.run()

def process_board(save_path):
    board = Board(save_path)
    board.run()

if __name__ == '__main__':
    save_path = 'data/'
    p_camera = multiprocessing.Process(target=process_camera, args=(save_path,))
    p_board = multiprocessing.Process(target=process_board, args=(save_path,))
    p_camera.start()
    board_start_time = time.perf_counter()
    p_board.start()
    gap = time.perf_counter() - board_start_time
    p_camera.join()
    p_board.join()
    save_file = save_path + 'board_timestamps.pickle' # Sync the timestamps
    board_timestamps = pickle.load(open(save_file, 'rb'))
    board_timestamps = [t + gap for t in board_timestamps]
    pickle.dump(board_timestamps, open(save_file, 'wb'))
    