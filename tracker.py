import time
import numpy as np
import cv2
from skimage.measure import label

class Tracker():
    BINARY_THRESHOLD = 180
    MIN_BINARY_THRESHOLD = 100
    AREA_MIN_THRESHOLD = 15
    AREA_MAX_THRESHOLD = 1000
    DIST_PER_SEC = 2000
    FOCAL_LENGTH_D435 = 1.021 # FOV = 91.2
    WIDTH_D435 = 800.0
    HEIGHT_D435 = 100.0
    BASELINE_D435 = 5.0 # The distance between the two infrared cameras
    REDUNDANT_PIXEL = 200

    def __init__(self, fps = 300):
        self.fps = fps
        self.Lx = self.Ly = -1
        self.Rx = self.Ry = -1
        self.pos_x = self.pos_y = self.pos_z = -1
        self.illuL = np.zeros((1,1))
        self.illuR = np.zeros((1,1))
        self.max_dist = 0
    
    def _track(self, image, lastX, lastY):
        center_image = image[:,self.REDUNDANT_PIXEL:int(self.WIDTH_D435)-self.REDUNDANT_PIXEL]
        center_image = np.resize(center_image, (int(np.shape(center_image)[0]//10), int(np.shape(center_image)[1]//10)))
        max_brightness = np.max(center_image)
        threshold = int(max(self.BINARY_THRESHOLD * max_brightness / 255, self.MIN_BINARY_THRESHOLD))
        _, binImg = cv2.threshold(image, threshold, 255, type=cv2.THRESH_BINARY)
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
                if ((lastX == -1 and lastY == -1) or dist * self.fps <= self.DIST_PER_SEC) and (self.REDUNDANT_PIXEL <= x and x < int(self.WIDTH_D435) - self.REDUNDANT_PIXEL) :
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
    
    def _calc_location_3D(self):
        if self.Lx != -1 and self.Ly != -1 and self.Rx != -1 and self.Ry != -1:
            self.pos_z = self.BASELINE_D435 / self.FOCAL_LENGTH_D435 / ((self.Lx - self.Rx) / (self.WIDTH_D435 / 2))
            self.pos_x = (self.Lx / (self.WIDTH_D435 / 2) - 1) * self.FOCAL_LENGTH_D435 * self.pos_z - (self.BASELINE_D435 / 2)
            self.pos_y = -(self.Ly / (self.HEIGHT_D435 / 2) - 1) * (self.HEIGHT_D435 / self.WIDTH_D435) * self.FOCAL_LENGTH_D435 * self.pos_z
        else:
            self.pos_x = -1
            self.pos_y = -1
            self.pos_z = -1
    
    def get_location(self):
        return [self.pos_x, self.pos_y, self.pos_z]

    def update(self, imgL, imgR):
        self.Lx, self.Ly = self._track(imgL, self.Lx, self.Ly)
        self.Rx, self.Ry = self._track(imgR, self.Rx, self.Ry)
        self._calc_location_3D()
        self.illuL = self._draw(imgL, self.Lx, self.Ly)
        self.illuR = self._draw(imgR, self.Rx, self.Ry)

if __name__ == '__main__':
    tracker = Tracker()

    camera_cap = cv2.VideoCapture('data/gyz-5/' + 'camera.avi')
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
            y = int(99 - (tracker.pos_y * 30 + 50))
            y = max(0,min(99,y))
            illu[:,:800-1] = illu[:,1:]
            illu[:,800-1] = 0
            illu[y,800-1] = 255
        cv2.imshow('illu', illu)

        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
