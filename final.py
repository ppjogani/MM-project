# -*- coding: utf-8 -*-
"""

@author: parin and sneha
"""

import numpy as np
import wave, struct
#import pylab as pl
from scipy import cluster as clu
#import matplotlib.colors as clrs
from skimage import color as cl

def calc_average(arr, start, end):
    #print "inside"
    average=0
    average=np.sum(arr)/(end-start)
    return average
  
def calc_yuv(yuv,merged_frame):
    addY=np.zeros(n)
    addition=np.zeros(n)
    for i in range(n):
        yuv[i,:,:]= merged_frame[i,:,:,0]*0.299 + merged_frame[i,:,:,1]*0.587 + merged_frame[i,:,:,2]*0.114
        addY[i]=np.sum(yuv[i,:,:].flatten(),dtype='int32')
    addition=np.abs(np.diff(addY,1))
    return yuv, addY, addition

def calc_hsv(merged_frame):
    hsv=np.zeros_like(merged_frame)
    hsv_sum=np.zeros(n)
    h_diff=np.zeros_like(hsv_sum)
    for t in range(n):
        hsv[t,:,:,:]=(cl.rgb2hsv(merged_frame[t,:,:,:]))
        hsv_sum[t]=np.sum(hsv[t,:,:,1])
    h_diff=np.abs(np.diff(hsv_sum,1))
    return hsv, hsv_sum, h_diff


def calc_db(slot_size,left,slots_y):
    ################calculate dBs again to check add and video
    chunks=[]
    for k in np.arange(1, slot_size):
        chunks.append(left[slots_y[k-1]*24000:(slots_y[k]-1)*24000])
    
    dbs_final = np.array([20*np.log10( np.sqrt(np.mean(chunk**2)) ) for chunk in chunks])
    return dbs_final
    
def numpy2string(y):
    """Expects a numpy vector of numbers, outputs a string"""
    signal = "".join((wave.struct.pack('h', item) for item in y))
    # this formats data for wave library, 'h' means data are formatted
    # as short ints
    return signal   
    
    
##########read file
print 'done'
#mat1=np.zeros(n, dtype='uint8')
mat1=np.fromfile("C:\\Users\\parin\\Documents\\MM\\video3-wreckitralph-full.rgb", "uint8", -1)
print "done"
#########3reshape
n=np.size(mat1)/(352*288*3)
mat_new=np.zeros([n, 288, 352, 3], dtype='uint8')
mat_new=np.transpose(np.reshape(mat1,[n,3,288,352]), [0,2,3,1])

merged_frame=np.zeros([n/12,288,352,3])
n=n-n%12
    
for i in np.arange(0, n , 12):
    merged_frame[i/12,:,:,0]=np.sum(mat_new[i:i+12,:,:,2],0,np.int16)/12
    merged_frame[i/12,:,:,1]=np.sum(mat_new[i:i+12,:,:,1],0,np.int16)/12
    merged_frame[i/12,:,:,2]=np.sum(mat_new[i:i+12,:,:,0],0,np.int16)/12
merged_frame=np.uint8(merged_frame)


n=n/12
yuv=np.array(np.zeros_like(merged_frame[:,:,:,0], np.float16))

yuv, addY, addition =calc_yuv(yuv, merged_frame)


hsv, hsv_sum, h_diff=calc_hsv(merged_frame)
#pl.plot(h_diff),pl.show()

#average_sh=np.zeros(n)


########make y shots
slots_yuv=np.array(np.where(addition>(np.max(addition)*.20))).T
slots_yuv=np.append(slots_yuv,n-1)
if slots_yuv[0]!=0:
    slots_yuv=np.append([0],slots_yuv)

#########make saturation shots
slots_hsv=np.array(np.where(h_diff>(np.max(h_diff)*.20))).T


##############combine
slots_y=np.unique(np.concatenate((slots_yuv,slots_hsv[:,0])))

######delete near by slots
slot_size=np.size(slots_y)
slots_del=np.zeros(slot_size)
h=0
for j in range(1,slot_size):
    if slots_y[j]-slots_y[j-1]==1:
        slots_del[h]=j
        h+=1

slots_y=np.delete(slots_y,slots_del[0:h],0)


######calc new slot size
slot_size=np.size(slots_y)

###############remove some slots
threshold=np.round(n*.008)
slot_size=np.size(slots_y)
for c in np.arange(1,slot_size):
    if np.abs(slots_y[c]-slots_y[c-1])<=threshold:
        slots_y[c]=slots_y[c-1]

slots_y=np.append(slots_y,n-1)        
slots_y=np.unique(slots_y)


######delete near by slots
slot_size=np.size(slots_y)
slots_del=np.zeros(slot_size)
h=0
for j in range(1,slot_size):
    if slots_y[j]-slots_y[j-1]==1:
        slots_del[h]=j
        h+=1

slots_y=np.delete(slots_y,slots_del[0:h],0)



#####################wav file
waveFile = wave.open("C:\\Users\\parin\\Documents\\MM\\video3-wreckitralph-full(1).wav", 'rb')

(nchannels, sampwidth, framerate, nframes, comptype, compname) = waveFile.getparams()
print nchannels,sampwidth,framerate,nframes
frames = waveFile.readframes (nframes * nchannels)
out = struct.unpack_from ("%dh" % nframes * nchannels, frames)
#left1 = np.abs(np.array(out))
waveFile.close()
left = np.array(out)
del out

slot_size=np.size(slots_y)
dbs=calc_db(slot_size,left,slots_y)

#chunks=[]
#for k in np.arange(1, slot_size):
#    chunks.append(left[slots_y[k-1]*24000:(slots_y[k]-1)*24000])
#
#dbs = np.array([20*np.log10( np.sqrt(np.mean(chunk**2)) ) for chunk in chunks])


dbs_size=np.size(dbs)

############################################fill the merged frame size with decibels to visualize
#########merge on decibels
#slots_del=np.zeros(dbs_size)
#h=0
#
#slot_dbs=np.zeros(n)
#for g in range(1,dbs_size):
#     slot_dbs[slots_y[g-1]:slots_y[g]]=dbs[g-1]   
#
##############take care of nan
#for f in np.arange(np.size(slot_dbs)):
#    if np.isnan(slot_dbs[f]):
#        slot_dbs[f]=slot_dbs[f+1]


#############delete slots on dB
thres=2.5
for j in range(1,dbs_size):
    if np.abs(dbs[j]-dbs[j-1])<=thres:        
        slots_del[h]=j
        h+=1

slots_y=np.delete(slots_y,slots_del[0:h-1],0)

############combine slots within 5 sec
#threshold=np.round(n*.01)
#slot_size=np.size(slots_y)
#for c in np.arange(1,slot_size):
#    if np.abs(slots_y[c]-slots_y[c-1])<=threshold:
#        slots_y[c]=slots_y[c-1]
#
#slots_y=np.append(slots_y,n-1)        
#slots_y=np.unique(slots_y)

################calculate dBs again to check add and video
slot_size=np.size(slots_y)
dbs_final=calc_db(slot_size,left,slots_y)
#chunks=[]
#for k in np.arange(1, slot_size):
#    chunks.append(left[slots_y[k-1]*24000:(slots_y[k]-1)*24000])
#
#dbs_final = np.array([20*np.log10( np.sqrt(np.mean(chunk**2)) ) for chunk in chunks])
dbs_size=np.size(dbs_final)
merge_t=2.5
for b in range(3):
    slots_new=np.copy(slots_y)
    for e in np.arange(1,dbs_size-1):
        if np.abs(dbs_final[e]-dbs_final[e-1])<=merge_t:
            if np.abs(dbs_final[e]-dbs_final[e+1])<=merge_t:
                if np.abs(dbs_final[e+1]-dbs_final[e-1])<=merge_t:
                    slots_y[e]=slots_y[e+2]
                    slots_y[e+1]=slots_y[e+2]
            else:
                slots_y[e]=slots_y[e+1]
    merge_t-=0.1

slots_y=np.unique(slots_y)
slot_size=np.size(slots_y)
dbs_final=calc_db(slot_size,left,slots_y)
    
###################weighted sum for decibels
#slot_size=np.size(slots_y)
#value=0
#for i in range(1,slot_size):
#    value+=(slots_y[i]-slots_y[i-1])*dbs_final[i-1]
#
#value=value/n
#
###################calc average for y
avgslots=np.zeros(slot_size-1)
for k in np.arange(1, slot_size):
    avgslots[k-1]=calc_average(addY[slots_y[k-1]:slots_y[k]], slots_y[k-1], slots_y[k]-1)

avgslots=avgslots/(352*288)

##############calc avg of saturation
avgslots_saturation=np.zeros(slot_size-1)
for k in np.arange(1, slot_size):
    avgslots_saturation[k-1]=calc_average(hsv_sum[slots_y[k-1]:slots_y[k]], slots_y[k-1], slots_y[k]-1)

avgslots_saturation=avgslots_saturation/(352*288)
avgslots_saturation_kmeans=avgslots_saturation/np.max(avgslots_saturation)


###############distinguish between advertisements and actual video
km_2=np.zeros(slot_size-1)
if np.size(np.where(avgslots_saturation_kmeans<=0.1))>1:
    for i in np.where(avgslots_saturation_kmeans<=0.1):
        km_2[i]=1
        
    km_2[19]=1
else:

    var_mean=np.zeros([slot_size-1],np.float32)
    var_R=np.zeros([slot_size-1],np.float32)
    
    for i in np.arange(1,slot_size):
        var_mean[i-1]=np.mean(np.var(yuv[slots_y[i-1]:slots_y[i],:,:],0,np.float32))
        var_R[i-1]=np.mean(np.var(merged_frame[slots_y[i-1]:slots_y[i],:,:,0],0,np.float32))
    
    var_sum=np.zeros([slot_size-1],np.float32)
    for i in np.arange(1,slot_size):
    #    var_var[i-1]=np.var(np.var(yuv[slots_y[i-1]:slots_y[i],:,:],0,np.float32))
        var_sum[i-1]=np.sum(np.var(yuv[slots_y[i-1]:slots_y[i],:,:],0,np.float32))
    
    
        
    avgslot_k=avgslots/np.max(avgslots)
    dbs_final_k=dbs_final/np.max(dbs_final)
    var_mean_k=var_mean/np.max(var_mean)
    #try_kmean=np.vstack([avgslot_k,dbs_final_k,var_mean_k]).T  
    try_kmean=np.vstack([avgslot_k,var_mean_k]).T  
    
    
    meadi=np.where(avgslot_k==np.sort(avgslot_k)[(slot_size-1)/2])[0][0]
    points=try_kmean[meadi-2:meadi]
        
    [km_1,km_2]=clu.vq.kmeans2(try_kmean,points,50,minit='points')

slot_sizes=np.zeros(slot_size-1)
for i in np.arange(slot_size-1):
    slot_sizes[i]=slots_y[i]-slots_y[i+1]

some_slots=np.where(slot_sizes>.25*n)[0]
some_size=np.size(some_slots)
if some_size>0:
    for j in some_slots:
        km_2[j]=0

size_1=0
size_2=0
for i in range(slot_size-1):
    if km_2[i]==0:
      size_1+=(slots_y[i+1]-slots_y[i])  
    elif km_2[i]==1:
      size_2+=(slots_y[i+1]-slots_y[i])

if size_1 < size_2:
    remove=0
else:
    remove=1
    
for i in range(slot_size-1):
    if km_2[i]==remove:
        mat1=np.delete(mat1,np.arange(slots_y[i]*12*352*288*3,slots_y[i+1]*12*352*288*3))
        left=np.delete(left,np.arange(slots_y[i]*24000,slots_y[i+1]*24000))

f=open('C:\\Users\\parin\\Documents\\MM\\testVideo3_ironman(new).rgb', 'wb')       
mat1.tofile(f)
f.close()


noise_output = wave.open('C:\\Users\\parin\\Documents\\MM\\testVideo3_ironman(new).wav', 'wb')
noise_output.setparams(waveFile.getparams())
noise_output.writeframes(numpy2string(left))
noise_output.close()



#var1=np.array([],dtype=np.uint8)
#var2=np.array([],dtype=np.uint8)
#for i in np.arange(1,slot_size):
#    if km_2[i-1]==0:
#        var1=np.append(var1,np.uint8(mat1[slots_y[i-1]*12*352*288*3:slots_y[i]*12*352*288*3]))
#    elif km_2[i-1]==1:
#        var2=np.append(var2,np.uint8(mat1[slots_y[i-1]*12*352*288*3:slots_y[i]*12*352*288*3]))
#var1_wav=np.array([],dtype=np.int32)
#for i in np.arange(1,slot_size):
#    if km_2[i-1]==0:
#        var1_wav=np.append(var1_wav,left[slots_y[i-1]*24000:slots_y[i]*24000])
#
#
