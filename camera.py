import pyrealsense2 as rs
import numpy as np
import cv2
import time
import _thread

def illustration(images, output_stream,is_running):
    cnt = 0
    while True:
        if cnt < len(images):
            output_stream.write(images[cnt])
            if cnt > 0:
                images[cnt-1] = None
            cnt += 1
        else:
            time.sleep(0.001)
        if cnt % 20 == 1:
            cv2.imshow('RealSense', images[-1])
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                is_running[0] = False
                break

if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 848, 100, rs.format.y8, 300)
    config.enable_stream(rs.stream.infrared, 2, 848, 100, rs.format.y8, 300)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.laser_power, 0)

    images = []
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output_stream = cv2.VideoWriter('video.avi', fourcc, 30, (848, 200), 0)
    is_running = [True]
    _thread.start_new_thread(illustration, (images, output_stream, is_running))

    t = time.time()
    last_time_gap = 0
    while is_running[0]:
        time_start = time.time_ns()
        frames = pipeline.wait_for_frames()
        time_gap = time.time_ns() - time_start
        if time_gap < last_time_gap: # waiting for sync
            frame0 = frames.get_infrared_frame(1)
            frame1 = frames.get_infrared_frame(2)
            image = np.vstack([np.asanyarray(frame0.get_data()), np.asanyarray(frame1.get_data())])
            images.append(image.copy())
        last_time_gap = time_gap
    
    print('Time = %.3f, FPS = %.1f' % (time.time() - t, len(images) / (time.time() - t)))
    output_stream.release()
