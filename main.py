import multiprocessing
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
    p_board.start()
    