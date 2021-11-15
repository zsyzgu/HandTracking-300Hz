import numpy as np
import sys
import time
import cv2
import _thread
import sensel

class Board():
    R = 105
    C = 185

    def __init__(self):
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
        self.output_stream = cv2.VideoWriter('video.avi', fourcc, 125, (self.C, self.R), 0)

    def _closeSensel(self):
        error = sensel.freeFrameData(self.handle, self._frame)
        error = sensel.stopScanning(self.handle)
        error = sensel.close(self.handle)
    
    def illustration(self):
        cnt = 0
        while self.is_running:
            if cnt < len(self.force_arrays):
                self.output_stream.write(self.force_arrays[cnt])
                if cnt > 0:
                    self.force_arrays[cnt-1] = None
                cnt += 1
            else:
                time.sleep(0.001)
            
            if cnt % 4 == 1:
                cv2.imshow('ForceArray', self.force_arrays[-1])
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    self.is_running = False
                    break

    def run(self):
        self.is_running = True
        _thread.start_new_thread(self.illustration, ())

        while (self.is_running):
            error = sensel.readSensor(self.handle)
            (error, num_frames) = sensel.getNumAvailableFrames(self.handle)
            for i in range(num_frames):
                error = sensel.getFrame(self.handle, self._frame)
            f = self._frame.force_array[:self.R*self.C]
            force_array = np.reshape(f, (self.R, self.C))
            self.force_arrays.append(force_array)

            for i in range(self._frame.n_contacts):
                c = self._frame.contacts[i]
                #print(c.id, c.state, c.x_pos, c.y_pos, c.area, c.total_force, c.major_axis, c.minor_axis)
        
        self._closeSensel()
        self.output_stream.release()

if __name__ == "__main__":
    board = Board()
    board.run()
