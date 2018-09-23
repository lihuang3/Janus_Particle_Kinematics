#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 12:53:08 2018

@author: lhuang28
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

def gaussian_fit(xdata,ydata):
    mu = np.sum(xdata*ydata)/np.sum(ydata)
    sigma = np.sqrt(np.abs(np.sum((xdata-mu)**2*ydata)/np.sum(ydata)))
    return mu, sigma

sum_offset = []
sum_vel = []

data_path = ['./janus2d/ncd1.csv','./janus2d/ncd2.csv', './janus2d/ncd3.csv']
for path in data_path:
    f = open(path, 'r')

    reader = csv.reader(f, delimiter=',')
    # get header from first row
    headers = next(reader)
    # get all the rows as a list
    data = list(reader)
    data  = [item[2:7] for item in data if  (float(item[0])>5 and float(item[3])>0 and float(item[4])>0)]
    data = np.array(data).astype(float)
    print len(data)
        
    gap = 10
    vel_thresh = 1.5
    dt10 = data[gap:,0]-data[:-gap,0]
    vel10 = np.sqrt(np.square(data[gap:,1]-data[:-gap,1])+np.square(data[gap:,2]-data[:-gap,2]))
    vel10 = vel10/dt10
      
    vel10 = vel10/26.*4.
    vel10f = vel10[vel10<vel_thresh]   #1.5
    if len(sum_vel)==0:
        sum_vel = vel10f
    else:
        sum_vel= np.append(sum_vel, vel10f)
    
    angle = data[gap:,-1]
    angle = angle[vel10<vel_thresh]
    
    offset = 180./np.pi*np.arctan2(data[gap:,2]-data[:-gap,2], data[gap:,1]-data[:-gap,1])
    offset = offset[vel10<vel_thresh]
    offset = offset-angle
    offset[offset<0] += 360
    offset[offset>280] -= 360
    if len(sum_offset)==0:
        sum_offset = offset
    else:
        sum_offset = np.append(sum_offset, offset)
    f.close()

mu, std = norm.fit(sum_offset)
#ydata = 1/np.sqrt(2*std*std*np.pi)*np.exp(-(sum_offset-mu)*(sum_offset-mu)/(2*std*std))    
#mu, std = gaussian_fit(sum_offset,ydata)

xmin, xmax = mu-90,mu+90

plt.figure(figsize = (30,20))


plt.hist(np.asarray(sum_offset), normed=True, bins=100)


x = np.linspace(xmin, xmax, 1000)
p = norm.pdf(x, mu+2.5, std/1.414)
print mu, std
plt.plot(x, p, 'k', linewidth=8)


plt.xlim(xmin, xmax)

plt.xlabel("Offset angle (degrees)", fontsize=80)
#plt.xlabel("Velocity magnitude ($\mu m/s$)", fontsize=50)
plt.ylabel('Density', fontsize=80)
plt.xticks(fontsize=80)
plt.yticks([],fontsize=80)
plt.savefig('ncdoffset.png')

mu, std = norm.fit(sum_vel)

xmin, xmax = mu-1.5,mu+1.5

plt.figure(figsize = (30,20))

plt.hist(np.asarray(sum_vel), normed=True, bins=50)

x = np.linspace(xmin, xmax, 1000)
p = norm.pdf(x, mu, std)
print mu, std
plt.plot(x, p, 'k', linewidth=8)

plt.xlim(xmin, xmax)
plt.xlabel("Velocity magnitude ($\mu m/s$)", fontsize=80)
plt.ylabel('Density', fontsize=80)
plt.xticks(fontsize=80)
plt.yticks([],fontsize=80)
plt.savefig('ncdvel.png')




#    plt.show()  
#    plt.figure(figsize = (30,20))
#    plt.hist(np.asarray(vel), normed=True, bins=50)
#    plt.xlabel("Velocity Magnitude ($\mu m/s$)", fontsize=38)
#    plt.ylabel('Density', fontsize=38)
#    plt.xticks([0, 1.0,2.0,3.0,4.0,5.0], fontsize=36)
#    plt.yticks(fontsize=36)
#
##    plt.show()
#    plt.savefig('vel_profile.png')



   
#    plt.figure(figsize = (30,20))
#    plt.hist(np.asarray(vel10), normed=False, bins=50)
#    plt.show()    