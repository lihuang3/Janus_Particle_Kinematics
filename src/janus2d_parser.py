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
class Detection():

    def __init__(self, vid_file):

        vidName = vid_file
        self.vid = av.open(vidName)
        self.num_frames = 0

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        	# cv2.createbackgroundSubstractor() works in cv3.0
        	# for 2.4.x use the following:
        self.fgbg = cv2.BackgroundSubtractorMOG2()
        self.detector = cv2.SimpleBlobDetector()
        

    def predict(self, frame):
        frameGray =  cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        thresh = 200 #175
        frameGray[frameGray>thresh] = 255

        self.fgmask = self.fgbg.apply(frameGray)
        self.fgmask[self.fgmask>220] = 0
        self.fgmask1 = self.fgmask
       
        #  Not good for small particles
        kernel = np.ones((2,2),np.uint8)
        self.fgmask = cv2.erode(self.fgmask, kernel, iterations = 1)
        kernel = np.ones((4,4),np.uint8)

        self.fgmask = cv2.dilate(self.fgmask,kernel,iterations = 1)

        self.fgmask = cv2.morphologyEx(self.fgmask, cv2.MORPH_OPEN, self.kernel)
#        cv2.imshow('1', self.fgmask)
#        cv2.waitKey(1)

        contours, _ = cv2.findContours(self.fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        positions = []
        
        if len(contours)>0:
            for contour in contours:
                cnt = contour
                area = cv2.contourArea(cnt)
                x,y,w,h = cv2.boundingRect(cnt)               
#                if area>256 and area<1024:
                if area>1024 and area<2048:  
                    positions.append([x,y,w,h])                    
        return positions
        
class Tracker():
    def __init__(self):
        self.history = []
        self.tracks = []
        self.next_id = 0
        self.max_age = 12
                
    def update(self, positions):                
        candidates = np.reshape(positions, [-1,4])

        for track in self.tracks:            
#            print "track bbox %s"%(str(track.bbox))
            candidates = track.update(candidates) 
            
        for i in range(np.shape(candidates)[0]):
            self.init_track(candidates[i,:])
              
        for track in self.tracks:
            if track.state == 0 and track.time_since_update>0:
                self.tracks.remove(track)
            elif track.time_since_update>self.max_age:                
                self.history.append(track)
                self.tracks.remove(track)

    
    def init_track(self, bbox):
        self.next_id += 1
        track = Track(bbox, self.next_id)                        
        self.tracks.append(track)    
    
        
def iom(bbox, candidates):

    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    _iou = area_intersection / (area_bbox + area_candidates - area_intersection)
    _iom = area_intersection / np.minimum(area_bbox, area_candidates)

    return _iou
    
class Track():
    def __init__(self, bbox, Id):        
        self.id = Id
        self.state = 0
        self.bbox = bbox
        self.hits = 1
        self.time_since_update = 0
        self.ninit = 3
        self.dt = 1./30.
        
    def update(self, candidates):
        if np.shape(candidates)[0]>0:
            iom_arr = iom(self.bbox, candidates)
            max_match = np.amax(iom_arr)
            if max_match>=0.5:
                index = np.where(iom_arr == max_match)
                index = np.squeeze(index)
                self.bbox = candidates[index,:]

                candidates = np.delete(candidates, index, 0)
                self.time_since_update = 0
                self.hits += 1                                  
                
                if self.state == 0 and self.hits>=self.ninit:
                    self.state = 1                            
            else:
                self.time_since_update+=1
        else:
            self.time_since_update+=1
        
        return candidates
               
                        
def pos2file(pos_file, frame_id, positions):
    pos_file.write("%d %d "%(frame_id, len(positions)))
    for item in positions:
        pos_file.write("%d %d %d %d "%(item[0],item[1], item[2], item[3]))
    pos_file.write("\n")    

def traj2file(pos_file, trajectory):
    for obj in trajectory:
        if len(obj)>300:
            for waypoint in obj:
                pos_file.write("%.2f "%(waypoint))
            pos_file.write("\n")

def bbox_dist(bbox1, bbox2):
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2
    return np.sqrt( np.square(x1+w1/2.-x2-w2/2.) +  np.square(y1+h1/2.-y2-h2/2.) )

def main():
    vid_file = './janus2d/ncd1.mp4'


#    './janus2d/test5_no_nacl_10__3um_coverSlip_largechamber_2018-09-02-163935-0000.mp4'
#    './janus2d/test4_no_nacl_10__3um_coverSlip_largechamber_2018-09-02-163646-0000.mp4'
#    './janus2d/test3_no_nacl_10__3um_coverSlip_largechamber_2018-09-02-163442-0000.mp4'
#    './janus2d/test2_no_nacl_10__3um_coverSlip_largechamber_2018-09-02-163051-0000.mp4'
#    './janus2d/test1_no_nacl_10__3um_coverSlip_largechamber_2018-09-02-162305-0000.mp4'
    print 'Processing video file %s'%(vid_file)    
    det_obj = Detection(vid_file)
    tracker = Tracker()
    pos_file = open('test1_data.txt', mode='w')
#    vel_file = open('vel_data.txt', mode='w')
    
    trajectory = []
    
    
    k = 0
    for frame_obj in det_obj.vid.decode(video=0):
        # Process a frame and output runners' coordinates and the resultant frame with detection
        # I = numpy.asarray(PIL.Image.open('test.jpg'))
        # im = PIL.Image.fromarray(numpy.uint8(I))
        frame = np.array(frame_obj.to_image())
        frame = cv2.resize(frame, (640, 512))
  
        k += 1
        
        print '\rframe %d'%(k),
        sys.stdout.flush() 
        
        if k>=30:

            positions = det_obj.predict(frame)

            tracker.update(positions)
            for bbox in positions:
                x,y,w,h = bbox                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)

            for track in tracker.tracks:
                x,y,w,h = track.bbox
                while len(trajectory)<track.id:
                    trajectory.append([])
                trajectory[track.id-1].extend([x+w/2., y+h/2.])
                
                if track.state == 0 or (track.state == 1 and track.time_since_update >1):
                    continue
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(frame,str(track.id), (int(x+w+5),int(y+h+5)),0, 5e-3 * 100, (0,255,0),1)
                             
                line_pts = np.int32(np.reshape(trajectory[track.id-1],[-1,2]) )
                
                for i in range(np.shape(line_pts)[0]-1):
                    cv2.line(frame, (line_pts[i][0],line_pts[i][1]), (line_pts[i+1][0],line_pts[i+1][1]),
                             color_dict[(track.id-1)%9],  1)  
            if k == 3447:
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
                    xest.append(xdata[0])
                    yest.append(ydata[0])
                    for i in range(1,np.shape(xdata)[0], gap):
                        tmpX = xest[-1]+4.66*10./30.*np.cos(0.867+np.pi*angle[i]/180.)
                        tmpY = yest[-1]+4.66*10./30.*np.sin(0.867+np.pi*angle[i]/180.)
                        xest.append(tmpX)
                        yest.append(tmpY)
            
                    for i in range(0,np.shape(xdata)[0]-1):
                        cv2.line(frame, (int(xdata[i]),int(ydata[i])), (int(xdata[i+1]),int(ydata[i+1])),
                             color_dict[(1)%9],  2) 
                    for i in range(0,np.shape(xest)[0]-35):
                        cv2.line(frame, (int(xest[i]),int(yest[i])), (int(xest[i+1]),int(yest[i+1])),
                             color_dict[(0)%9],  2)
            
            if k==3447:
                cv2.imshow('final_frame', frame) 
                cv2.imwrite('overlap1.jpg',frame)
                cv2.waitKey(0)
#    traj2file(pos_file, trajectory)                     
    pos_file.close()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
