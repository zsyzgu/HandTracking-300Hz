import cv2
import numpy as np

def video2images(videoPath, imagePath):
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    suc = cap.isOpened()
    frameCnt = 0
    while suc:
        print(frameCnt)
        frameCnt += 1
        suc, frame = cap.read()
        if suc:
            cv2.imwrite(imagePath + '%04d.jpg' % frameCnt, frame)
            cv2.waitKey(1)
    cap.release()

if __name__ == '__main__':
    #video2images('data.mp4', 'data/')

    N = 500
    images = []
    for i in range(1,N):
        image = cv2.imread('data/' + '%04d.jpg' % i)
        images.append(image)
    
    print(len(images))
    for i in range(1,N):
        H, W, C = images[i].shape
        prvs = cv2.cvtColor(images[i-1],cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)

        hsv = np.zeros_like(images[i])
        hsv[...,1] = 255
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 笛卡尔坐标转换为极坐标，获得极轴和极角
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        hsv[...,0] = ang*180/np.pi/2  #角度
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) 
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2',bgr)
        cv2.imshow("frame1", next)
        key = cv2.waitKey(30) & 0xff

