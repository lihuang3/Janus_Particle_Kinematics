#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:13:08 2018

@author: lhuang28
"""

import matplotlib.pyplot as plt
import numpy as np

vel_file = open('vel_data.txt','r')
vel_data = vel_file.readlines()
y = []
for id, line in enumerate(vel_data):    
    vel = line.split()
    if len(vel)>20:
        data = [float(i) for i in vel]
        for item in data:
            if item>0:
                y.append(item)




plt.hist(np.asarray(y), normed=False, bins=25)
plt.show()
    