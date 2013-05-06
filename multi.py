# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
C:\WinPython-64bit-2.7.3.3\settings\.spyder2\.temp.py
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
mat1=np.fromfile("C:\\WinPython-64bit-2.7.3.3\\python-2.7.3.amd64\\Scripts\\MM\\video2-messi-full.rgb", "uint8", -1)
print "done"
#########3reshape
n=np.size(mat1)/(352*288*3)
mat_new=np.zeros([n, 288, 352, 3], dtype='uint8')
mat_new=np.transpose(np.reshape(mat1,[n,3,288,352]), [0,2,3,1])
yuv=np.array(np.zeros_like(mat_new[0:n,:,:,0], np.float16))
del mat1
addY=np.zeros(n)
diffY=np.zeros(n)
difference=np.zeros(n)
addition=np.zeros(n)
for i in range(n):
    yuv[i,:,:]= mat_new[i,:,:,2]*0.299 + mat_new[i,:,:,1]*0.587 + mat_new[i,:,:,0]*0.114
    addY[i]=np.sum(yuv[i,:,:].flatten(),dtype='int32')
    #diffY[i]=np.sum(np.diff(yuv[i,:,:].flatten(),2),dtype='int32')
    if i != 1:
        addition[i]=np.abs(addY[i]-addY[i-1])
        #difference[i]=np.abs(diffY[i]-diffY[i-1])
#buckets=np.arange(0,255,22)
#353 pacific oaks road,

#pl.figure()
#pl.subplot(211)
#pl.plot(np.arange(n),difference)
#pl.subplot(212)
#pl.plot(np.arange(n),addition)
#pl.show()



#read audio file
waveFile = wave.open("C:\\WinPython-64bit-2.7.3.3\\python-2.7.3.amd64\\Scripts\\MM\\video2-messi-full(1).wav", 'rb')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = waveFile.getparams()
print nchannels,sampwidth,framerate,nframes
frames = waveFile.readframes (nframes * nchannels)
out = struct.unpack_from ("%dh" % nframes * nchannels, frames)
left = np.array(out)
#print np.count_nonzero(left)
#plot audio data 
x=np.size(left)
sum_size=abs(x/2000)
sum_wav=np.zeros(sum_size)
j=0
for i in range(0,sum_size):
    j=i*2000
    sum_wav[i]=np.sum(left[j:j+1999])
#y=np.size(left[2*x/3:])
#print 1
#pl.figure()
#pl.subplot(411)
#pl.plot(np.arange(x/3),left[:x/3])
#pl.subplot(412)
#pl.plot(np.arange(x/3),left[x/3:2*x/3])
#pl.subplot(413)
#pl.plot(np.arange(y),left[2*x/3:])
#pl.subplot(111)
#pl.plot(np.arange(sum_size),sum_wav)
#pl.show()

xx=addition>2200000
y=np.where(xx)
yy=np.zeros_like(y)
g=1
for b in np.arange(1,np.size(y)):
    if y[0][b]-y[0][b-1]<24:
        yy[0][g-1]=y[0][b-1]
    else:
        yy[0][g]=y[0][b]
        g+=1
yy.resize(g)
#y=np.insert(y,0,[0])
ysize=g
avgslots=np.zeros(ysize)
h=0
newyy=np.zeros_like(yy)
for k in np.arange(1, ysize):
    avgslots[h]=calc_average(sum_wav[yy[k-1]:yy[k]-1], yy[k-1], yy[k]-1)
    if h==0:
        newyy[h]=0
        h+=1
        continue
    if np.abs(avgslots[h]-avgslots[h-1])<6000:
        avgslots[h-1]=calc_average(sum_wav[yy[k-2]:yy[k]], yy[k-2], yy[k])
        newyy[h]=yy[k]
        h-=1
    else:
        newyy[h]=yy[k-1]
# avgslots[h]=np.sum(sum_wav[yy[k-1]:yy[k]])/(yy[k]-yy[k-1])
    h+=1    
h-=1

#back_video=np.zeros(h)
#diffvideo=np.zeros(h)
#diffaudio=np.zeros(h)
#for z in np.arange(1,h):
#    back_video[z]=calc_average(addY[newyy[z-1]:newyy[z]-1],newyy[z-1],newyy[z]-1)
#for p in np.arange(h):
#    diffvideo[p]=np.abs(back_video[p]-back_video[p-2])
#    diffaudio[p]=np.abs(avgslots[p]-avgslots[p-2])
    

#pl.figure()
#pl.subplot(421)
#p1 = 20*np.log10(np.abs(np.fft.rfft(left[0:888*2000])))
#f1 = np.linspace(0, framerate/2.0, len(p1))
#pl.plot(f1, p1)
#pl.subplot(422)
#p2 = 20*np.log10(np.abs(np.fft.rfft(left[955*2000:1632*2000])))
#f2 = np.linspace(0, framerate/2.0, len(p2))
#pl.plot(f2, p2)
#
#pl.subplot(423)
#p3 = 20*np.log10(np.abs(np.fft.rfft(left[1633*2000:5711*2000])))
#f3 = np.linspace(0, framerate/2.0, len(p3))
#pl.plot(f3, p3)
#pl.subplot(424)
#p4 = 20*np.log10(np.abs(np.fft.rfft(left[5712*2000:6240*2000])))
#f4 = np.linspace(0, framerate/2.0, len(p4))
#pl.plot(f4, p4)
#
#pl.subplot(425)
#p5 = 20*np.log10(np.abs(np.fft.rfft(left[6240*2000:8087*2000])))
#f5 = np.linspace(0, framerate/2.0, len(p5))
#pl.plot(f5, p5)
#pl.subplot(426)
#p6 = 20*np.log10(np.abs(np.fft.rfft(left[8088*2000:9215*2000])))
#f6 = np.linspace(0, framerate/2.0, len(p6))
#pl.plot(f6, p6)
#
#pl.subplot(427)
#p7 = 20*np.log10(np.abs(np.fft.rfft(left[9216*2000:])))
#f7 = np.linspace(0, framerate/2.0, len(p7))
#pl.plot(f7, p7)
#pl.show()



#pl.subplot(211)
#pl.plot(left[0:1632*2000])
#
#pl.subplot(212)
#[bin, freq, Pxx, im] = pl.specgram(left[0:6240*2000], Fs = framerate, NFFT=48000, noverlap=0, scale_by_freq=True,sides='default')
#pl.show()

hsv=np.zeros_like(mat_new)
hsv_sum=np.zeros(n)
for t in range(100):
    er=np.array([mat_new[t,:,:,2],mat_new[t,:,:,1],mat_new[t,:,:,0]])
    er1=np.transpose(er,[1,2,0])
    hsv[t,:,:,:]=cl.rgb2hsv(er1)
    hsv_sum[t]=np.sum(hsv[t,:,:,0])
