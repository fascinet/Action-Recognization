import numpy as np
import cv2 as cv
from os import *
import os
dir='video2'
for name in listdir(dir):
    path1=dir+'/'+name[:len(name)-4]+'/rgb'
    path2=dir+'/'+name[:len(name)-4]+'/opticalflow'
    print(path1,path2)
    if not os.path.exists(path1):
      cap = cv.VideoCapture(cv.samples.findFile(dir+'/'+name))
      os.makedirs(path1)
      os.makedirs(path2)
      ret, frame1 = cap.read()
      prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
      hsv = np.zeros_like(frame1)
      hsv[...,1] = 255
      val=0
      l=1
      while(1):
        ret, frame2 = cap.read()
        if ret:
            next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
            #cv.imshow('frame2',bgr)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == 255:
                name1=path1+'/'+str(l)+'.jpg'
                name2=path2+'/'+str(l)+'.jpg'
                print("Creating.."+name1+str(l))
                print("Creating.."+name2+str(l))
                cv.imwrite(name1,frame2)
                cv.imwrite(name2,bgr)
                l+=1
        else:
            break
      cap.release()
      cv.destroyAllWindows()
    else:
      print('Done')  