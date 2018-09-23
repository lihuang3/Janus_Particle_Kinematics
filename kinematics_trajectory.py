#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:07:45 2018

@author: lhuang28
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 11:12:38 2018

@author: lihuang
"""
import csv
import cv2, sys, os, av
import numpy as np
from scipy import spatial
import pdb

# RGB
#color_dict = {0:(255,0,0), 1:(255,255,0), 2:(0,0,255), 3:(148, 0, 211), 4: (255, 127, 0),
#                5:(0,255,0), 6:(75, 0, 130), 7:(255,255,255), 8:(0,0,0)}

color_dict = {0:(0,0, 255), 1:(0,255,255), 2:(255, 0,0), 3:(148, 0, 211), 4: (0, 127, 255),
                5:(0,255,0), 6:(130, 0, 75), 7:(255,255,255), 8:(0,0,0)}


def main():
    vid_file = './janus2d/ncd1.mp4'

    vid = av.open(vid_file)
    print 'Processing video file %s'%(vid_file)    
    
    with open('./janus2d/ncd1.csv','r') as f:
        reader = csv.reader(f, delimiter=',')
        # get header from first row
        headers = next(reader)
        # get all the rows as a list
        
        data = list(reader)
        data  = [item[2:7] for item in data if  (float(item[0])>5 and float(item[3])>0 and float(item[4])>0)]
        data = np.array(data).astype(float)
                   
        
        
        gap = 10
        vel_thresh = 1.5
        dt10 = data[gap:,0]-data[:-gap,0]
        vel10 = np.sqrt(np.square(data[gap:,1]-data[:-gap,1])+np.square(data[gap:,2]-data[:-gap,2]))
        vel10 = vel10/dt10
          
        vel10 = vel10/26.*4.
        vel10f = vel10[vel10<vel_thresh]   #1.5
         
        xdata = data[gap:,1]
        xdata = xdata[vel10<vel_thresh]
        ydata = data[gap:,2]
        ydata = ydata[vel10<vel_thresh]                                       
    
        angle = data[gap:,-1]
        angle = angle[vel10<vel_thresh]
        
        xest = []
        yest = []
        
        print 'raw data len = %d'%(len(data))
        print 'effective data len = %d'%(len(vel10))

        xest.append(xdata[0])
        yest.append(ydata[0])
        xdata1,ydata1= [],[]
        xdata1.append(xdata[0])
        ydata1.append(ydata[0])
        for i in range(1,np.shape(xdata)[0], gap):            
            tmpX = xest[-1]+4.5*gap/30.*np.cos(0.867+np.pi*angle[i]/180.)
            tmpY = yest[-1]+4.5*gap/30.*np.sin(0.867+np.pi*angle[i]/180.)
            xest.append(tmpX)
            yest.append(tmpY)
            xdata1.append(xdata[i])
            ydata1.append(ydata[i])
        
        _seq = np.asarray(range(50,3447))
        _seq = _seq[vel10<vel_thresh]
        vid_seq = [] 
        for i in range(1,np.shape(xdata)[0], gap):            
            vid_seq.append(_seq[i])

            
    k = 0
    w, h = 40,40
    detect_traj = []
    predict_traj = []
    idx = 0
    for frame_obj in vid.decode(video=0):
        
        frame = np.array(frame_obj.to_image())


        k += 1      
        print '\rframe %d'%(k),
        sys.stdout.flush()
        
        
        if k in vid_seq:
            x,y = xdata1[idx],ydata1[idx] 
            detect_traj.append([x,y])
            x_pred, y_pred = xest[idx], yest[idx]
            predict_traj.append([x_pred,y_pred])
            idx+=1
            
#            cv2.rectangle(frame,(int(x-w/2.),int(y-h/2.)),(int(x+w/2.),int(y+h/2.) ),(255,255,255),2)
#            cv2.rectangle(frame,(int(x_pred-w/2.),int(y_pred-h/2.)),(int(x_pred+w/2.),int(y_pred+h/2.) ),(0,0,255),2)
#            
#            detect_line = np.reshape(detect_traj,[-1,2]).astype(np.int32)
#            predict_line = np.reshape(predict_traj,[-1,2]).astype(np.int32)
#            print detect_traj
            for i in range(len(detect_traj)-1):
                cv2.line(frame, (int(detect_traj[i][0]),int(detect_traj[i][1])), (int(detect_traj[i+1][0]),int(detect_traj[i+1][1])),
                             color_dict[(7)%9],  2) 
                cv2.line(frame, (int(predict_traj[i][0]),int(predict_traj[i][1])), (int(predict_traj[i+1][0]),int(predict_traj[i+1][1])),
                             color_dict[(0)%9],  2) 


#            cv2.imshow('final_frame', frame) 
#            cv2.waitKey(1)
            cv2.imwrite('./ncd1frames/ncd1_%03d.png'%(idx),frame)



    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
