import pickle
import numpy as np
import time
import cv2
from threading import Thread
import sensel

class Board():
    R = 105
    C = 185

    def __init__(self, save_path):
        self.save_path = save_path
        (error, device_list) = sensel.getDeviceList()
        if device_list.num_devices != 0:
            (error, handle) = sensel.openDeviceByID(device_list.devices[0].idx)
        self.handle = handle
        (error, self.info) = sensel.getSensorInfo(self.handle)
        error = sensel.setFrameContent(self.handle, 0x05)
        error = sensel.setContactsMask(self.handle, 0x01)
        (error, frame) = sensel.allocateFrameData(self.handle)
        error = sensel.startScanning(self.handle)
        self._frame = frame
        self.force_arrays = []
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.output_stream = cv2.VideoWriter(save_path + 'board.avi', fourcc, 125, (self.C, self.R), 0)
        self.timestamps = []
        self.contacts = []

    def _closeSensel(self):
        error = sensel.freeFrameData(self.handle, self._frame)
        error = sensel.stopScanning(self.handle)
        error = sensel.close(self.handle)
    
    def _illustration(self):
        cnt = 0
        while self.is_running:
            if cnt < len(self.force_arrays):
                self.output_stream.write(self.force_arrays[cnt])
                if cnt > 0:
                    self.force_arrays[cnt-1] = None # release memory
                cnt += 1
            else:
                time.sleep(0.001)
            
            if cnt % 4 == 1:
                cv2.imshow('ForceArray', self.force_arrays[-1])
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    self.is_running = False
        
        self.recoarded = cnt

    def _save_data(self):
        print('[Board] Time = %.3f, FPS = %.1f' % (self.timestamps[-1] - self.timestamps[0], (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])))
        cnt = self.recoarded
        while cnt < len(self.force_arrays):
            self.output_stream.write(self.force_arrays[cnt])
            cnt += 1
        self.output_stream.release()
        pickle.dump(self.timestamps, open(self.save_path + 'board_timestamps.pickle', 'wb'))
        pickle.dump(self.contacts, open(self.save_path + 'board_contacts.pickle', 'wb'))

    def run(self):
        self.is_running = True
        thread_illu = Thread(target=self._illustration, args=())
        thread_illu.start()

        while (self.is_running):
            error = sensel.readSensor(self.handle)
            (error, num_frames) = sensel.getNumAvailableFrames(self.handle)
            for i in range(num_frames):
                error = sensel.getFrame(self.handle, self._frame)
            self.timestamps.append(time.perf_counter())
            force_array = self._frame.force_array[:self.R*self.C]
            force_array = np.clip((np.reshape(force_array, (self.R, self.C)) * 10),0,255).astype(np.uint8)
            self.force_arrays.append(force_array)

            frame_contacts = []
            for i in range(self._frame.n_contacts):
                c = self._frame.contacts[i]
                frame_contacts.append([c.id, c.state, c.x_pos, c.y_pos, c.area, c.total_force, c.major_axis, c.minor_axis])
            self.contacts.append(frame_contacts)
        
        thread_illu.join()
        self._closeSensel()
        self._save_data()

if __name__ == "__main__":
    board = Board('data/')
    board.run()