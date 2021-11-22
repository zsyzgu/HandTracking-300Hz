from numpy.lib.npyio import save
import serial
import time
import madgwickahrs
import math
import numpy as np
import os
import pickle
import cv2
from threading import Thread

class IMU():
    #9250
    offset_gx = -0.131
    offset_gy = -0.144
    offset_gz = +0.084

    def __init__(self, save_path):
        self.save_path = save_path
        self.ser = serial.Serial('COM5', 250000)
        time.sleep(1)
        self.ser.read(self.ser.in_waiting)
        self.mad = madgwickahrs.MadgwickAHRS(0.001)
        self.heading = 0
        self.data = []
    
    def _read_buf(self, data, offset):
        return int.from_bytes(data[offset:offset+2],byteorder='little',signed=True)

    def _get_data(self):
        while (self.mad == None):
            time.sleep(0.1)
        
        data = self.ser.read(12)
        timestamp = time.perf_counter()
        ax = self._read_buf(data, 0) / 4096.0
        ay = self._read_buf(data, 2) / 4096.0
        az = self._read_buf(data, 4) / 4096.0
        gx = self._read_buf(data, 6) / 65.5 * (math.pi / 180.0) - self.offset_gx
        gy = self._read_buf(data, 8) / 65.5 * (math.pi / 180.0) - self.offset_gy
        gz = self._read_buf(data, 10) / 65.5 * (math.pi / 180.0) - self.offset_gz
        self.mad.update_imu([gx, gy, gz], [ax, ay, az])
        [gra_x, gra_y, gra_z] = self.mad.calc_gravity()
        ax -= gra_x
        ay -= gra_y
        az -= gra_z
        gra_x = min(gra_x, +1)
        gra_x = max(gra_x, -1)
        pitch, heading = self.mad.calc_angles()
        self.heading += gz * 0.001
        return [timestamp, gx, gy, gz, ax, ay, az, gra_x, gra_y, gra_z, pitch, self.heading]
    
    def _illustration(self):
        image = np.zeros((100,100))

        cnt = 0
        while self.is_running:
            if cnt % 10 == 1:
                cv2.imshow('RealSense', image)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    self.is_running = False
            else:
                time.sleep(0.001)
            cnt += 1
            
    
    def _save_data(self):
        print('[IMU] Time = %.3f, FPS = %.1f' % (self.data[-1][0] - self.data[0][0], (len(self.data) - 1) / (self.data[-1][0] - self.data[0][0])))
        pickle.dump(self.data, open(self.save_path + 'imu_data.pickle', 'wb'))

    def calibration(self):
        print('Please make sure that the IMU is stationary.')
        os.system('pause')
        gx_array = []
        gy_array = []
        gz_array = []
        for i in range(1000):
            data = self.get_data()
            gx_array.append(data[0])
            gy_array.append(data[1])
            gz_array.append(data[2])
        print('offset_gx = ', np.mean(gx_array))
        print('offset_gy = ', np.mean(gy_array))
        print('offset_gz = ', np.mean(gz_array))

    def run(self):
        self.is_running = True

        thread_illu = Thread(target=self._illustration, args=())
        thread_illu.start()
        
        while self.is_running:
            self.data.append(self._get_data())
        
        thread_illu.join()
        self._save_data()

if __name__ == '__main__':
    imu = IMU('data/')
    imu.run()
