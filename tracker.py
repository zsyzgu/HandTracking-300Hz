import time
import numpy as np
import cv2
from skimage.measure import label

class Tracker():
    BINARY_THRESHOLD = 110
    AREA_MIN_THRESHOLD = 20
    AREA_MAX_THRESHOLD = 1000
    DIST_PER_SEC = 2000

    def __init__(self, fps = 300):
        self.fps = fps
        self.Lx = self.Ly = -1
        self.Rx = self.Ry = -1
        self.illuL = np.zeros((1,1))
        self.illuR = np.zeros((1,1))
        self.max_dist = 0
    
    def _track(self, image, lastX, lastY):
        _, binImg = cv2.threshold(image, self.BINARY_THRESHOLD, 255, type=cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_size = 0
        ans_x = -1
        ans_y = -1
        for contour in contours:
            size = cv2.contourArea(contour)
            if self.AREA_MIN_THRESHOLD <= size and size <= self.AREA_MAX_THRESHOLD and size > max_size:
                M = cv2.moments(contour)
                x = M['m10'] / M['m00']
                y = M['m01'] / M['m00']
                dist = ((x - lastX) ** 2 + (y - lastY) ** 2) ** 0.5
                if (lastX == -1 and lastY == -1) or dist * self.fps <= self.DIST_PER_SEC:
                    max_size = size
                    ans_x = x
                    ans_y = y
        return ans_x, ans_y
    
    def _draw(self, image, x, y):
        output = image.copy()
        x = int(x)
        y = int(y)
        output = cv2.line(output, (x-3,y), (x+3,y), color=(0,0,0), thickness=1)
        output = cv2.line(output, (x,y-3), (x,y+3), color=(0,0,0), thickness=1)
        return output

    def update(self, imgL, imgR):
        self.Lx, self.Ly = self._track(imgL, self.Lx, self.Ly)
        self.Rx, self.Ry = self._track(imgR, self.Rx, self.Ry)
        self.illuL = self._draw(imgL, self.Lx, self.Ly)
        self.illuR = self._draw(imgR, self.Rx, self.Ry)

if __name__ == '__main__':
    tracker = Tracker()

    camera_cap = cv2.VideoCapture('data/gyz-6/' + 'camera.avi')
    camera_frames = []
    illu = np.zeros((100,800))
    cnt = 0
    while True:
        cnt += 1
        ret, frame = camera_cap.read()
        if frame is None:
            break
        if cnt < 1000:
            continue
        camera_frames.append(frame)
        H, W, depth = np.shape(frame)
        imgL = frame[:H//2,:,0]
        imgR = frame[H//2:,:,0]
        tracker.update(imgL, imgR)
        cv2.imshow('L', tracker.illuL)
        cv2.imshow('R', tracker.illuR)

        if tracker.Ly != -1:
            y = int((tracker.Ly + tracker.Ry) / 2)
            y = max(0,min(99,y))
            illu[:,:800-1] = illu[:,1:]
            illu[:,800-1] = 0
            illu[y,800-1] = 255
        cv2.imshow('illu', illu)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
