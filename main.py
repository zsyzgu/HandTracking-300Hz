import pyrealsense2 as rs
import numpy as np
import cv2
import time
import _thread

def illustration(images):
    cnt = 0
    while True:
        if cnt + 20 <= len(images):
            cv2.imshow('RealSense', images[-1])
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            cnt = len(images)
        time.sleep(0.005)

if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 848, 100, rs.format.y8, 300)
    config.enable_stream(rs.stream.infrared, 2, 848, 100, rs.format.y8, 300)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.laser_power, 0)

    images = []
    _thread.start_new_thread(illustration, (images,))

    t = time.time()
    for i in range(2000):
        frames = pipeline.wait_for_frames()
        frame0 = frames.get_infrared_frame(1)
        frame1 = frames.get_infrared_frame(2)

        if frame0 and frame1:
            image = np.vstack([np.asanyarray(frame0.get_data()), np.asanyarray(frame1.get_data())])
            images.append(image.copy())
    
    print(time.time() - t, len(images))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('video.avi', fourcc, 10, (848, 200), 0)

    for i in range(len(images)):
        image = images[i]
        out.write(image)

    out.release()
