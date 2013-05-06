# -*- coding: utf-8 -*-
"""
Created on Sun May 05 10:29:43 2013

@author: Parin
"""

import numpy as np
import wave, struct
import pylab as pl
#import matplotlib.colors as clrs
from skimage import color as cl

def calc_average(arr, start, end):
    #print "inside"
    average=0
    average=np.sum(arr)/(end-start)
    return average
  
    
##########read file
print 'done'
#mat1=np.zeros(n, dtype='uint8')
mat1=np.fromfile("C:\\WinPython-64bit-2.7.3.3\\python-2.7.3.amd64\\Scripts\\MM\\video3-wreckitralph-full.rgb", "uint8", -1)
print "done"
#########3reshape
n=np.size(mat1)/(352*288*3)
mat_new=np.zeros([n, 288, 352, 3], dtype='uint8')
mat_new=np.transpose(np.reshape(mat1,[n,3,288,352]), [0,2,3,1])
del mat1
merged_frame=np.zeros([n/12,288,352,3])
n=n-n%12
    
for i in np.arange(0, n , 12):
    merged_frame[i/12,:,:,0]=np.sum(mat_new[i:i+12,:,:,2],0,np.int16)/12
    merged_frame[i/12,:,:,1]=np.sum(mat_new[i:i+12,:,:,1],0,np.int16)/12
    merged_frame[i/12,:,:,2]=np.sum(mat_new[i:i+12,:,:,0],0,np.int16)/12
merged_frame=np.uint8(merged_frame)
del mat_new

n=n/12
yuv=np.array(np.zeros_like(merged_frame[:,:,:,0], np.float16))

addY=np.zeros(n)
#diffY=np.zeros(n)
#difference=np.zeros(n)
addition=np.zeros(n)
for i in range(n+1):
    yuv[i,:,:]= merged_frame[i,:,:,0]*0.299 + merged_frame[i,:,:,1]*0.587 + merged_frame[i,:,:,2]*0.114
    addY[i]=np.sum(yuv[i,:,:].flatten(),dtype='int32')
    #diffY[i]=np.sum(np.diff(yuv[i,:,:].flatten(),2),dtype='int32')
addition=np.abs(np.diff(addY,1))
#yuv=np.array(np.zeros_like(mat_new[0:n,:,:,0], np.float16))


hsv=np.zeros_like(merged_frame)
hsv_sum=np.zeros(n)
h_diff=np.zeros_like(hsv_sum)
for t in range(n+1):
#    er=np.array([merged_frame[t,:,:,2],mat_new[t,:,:,1],mat_new[t,:,:,0]]).
#    er1=np.transpose(er,[1,2,0])
    hsv[t,:,:,:]=cl.rgb2hsv(merged_frame[t,:,:,:])
    hsv_sum[t]=np.sum(hsv[t,:,:,1])
h_diff=np.abs(np.diff(hsv_sum,1))
#pl.imshow(merged_frame[140,:,:,:]),pl.show()
average_sh=np.zeros(n)
slots_y=np.append([0],np.array(np.where(addition>(np.max(addition)*.25))).T)
slot_size=np.size(slots_y)
for j in range(1,slot_size):
    average_sh[slots_y[j-1]:slots_y[j]]=calc_average(hsv[slots_y[j-1]:slots_y[j]],slots_y[j-1],slots_y[j])




pl.figure()
pl.subplot(211)
pl.plot(addition>(np.max(addition)*.25))
pl.subplot(212)
pl.plot(np.abs(np.diff(average_sh,0)))
pl.show()


