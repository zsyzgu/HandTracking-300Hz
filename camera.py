import pyrealsense2 as rs
import numpy as np
import cv2
import time
from threading import Thread
import pickle

class Camera():
    def __init__(self, save_path):
        self.save_path = save_path
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.infrared, 1, 848, 100, rs.format.y8, 300)
        config.enable_stream(rs.stream.infrared, 2, 848, 100, rs.format.y8, 300)

        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.laser_power, 0)

        self.images = []
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.output_stream = cv2.VideoWriter(save_path + 'camera.avi', fourcc, 30, (848, 200), 0)
        self.timestamps = []

    def _illustration(self):
        cnt = 0
        while self.is_running:
            if cnt < len(self.images):
                self.output_stream.write(self.images[cnt])
                if cnt > 0:
                    self.images[cnt-1] = None # release memory
                cnt += 1
            else:
                time.sleep(0.001)
            
            if cnt % 10 == 1:
                cv2.imshow('RealSense', self.images[-1])
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    self.is_running = False
        
        self.recorded = cnt
        
    def _save_data(self):
        print('[Camera] Time = %.3f, FPS = %.1f' % (self.timestamps[-1] - self.timestamps[0], (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])))
        cnt = self.recorded
        while cnt < len(self.images):
            self.output_stream.write(self.images[cnt])
            cnt += 1
        self.output_stream.release()
        pickle.dump(self.timestamps, open(self.save_path + 'camera_timestamps.pickle', 'wb'))

    def run(self):
        self.is_running = True
        thread_illu = Thread(target=self._illustration, args=())
        thread_illu.start()

        start_time = time.perf_counter()
        last_time_gap = 0
        while self.is_running:
            t = time.perf_counter()
            frames = self.pipeline.wait_for_frames()
            time_gap = time.perf_counter() - t
            if time_gap < last_time_gap: # waiting for sync
                self.timestamps.append(time.perf_counter())
                frame0 = frames.get_infrared_frame(1)
                frame1 = frames.get_infrared_frame(2)
                image = np.vstack([np.asanyarray(frame0.get_data()), np.asanyarray(frame1.get_data())])
                self.images.append(image.copy())
            last_time_gap = time_gap
        
        thread_illu.join()
        self._save_data()

if __name__ == "__main__":
    camera = Camera('data/')
    camera.run()
