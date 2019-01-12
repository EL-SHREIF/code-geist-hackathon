# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:53:24 2018

@author: shiko
"""

import tensorflow as tf
import numpy as np
import os,cv2
import sys,argparse
from glob import glob
import time
import ctypes

current = 0x00
old = 0x00

SendInput = ctypes.windll.user32.SendInput
W =  0x11
A= 0x1E
S = 0x1F
D = 0x20 

#=========================================================
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

#=========================================================
def releaseandswitch (hexKeyCode1,hexKeyCode2):
    ReleaseKey(hexKeyCode1)
    PressKey(hexKeyCode2)
    
def releaseandswitcho (hexKeyCode1,hexKeyCode2):
    ReleaseKey(hexKeyCode1)
    PressKey(hexKeyCode2)
#    time.sleep(.4)
    ReleaseKey(hexKeyCode2)
    



c_frame=-1
p_frame=-1

#Setting threshold for number of frames to compare
thresholdframes=50


## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('C:\\Users\\shiko\\Desktop\\New folder (2)\\handgest_1.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 10)) 


#Real Time prediction
def predict(frame,y_test_images):
    image_size=50
    num_channels=3
    images = []
    image=frame
    cv2.imshow('test',image)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)

    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)

    ### Creating the feed_dict that is required to be fed to calculate y_pred 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of gest0,......,probability_of_gest9]
    return np.array(result)

####TestData prediction
##testpath='C:/Users/Raj Shah/Downloads/cv-tricks.com-master/Tensorflow-tutorials/tutorial-2-image-classifier/Testdata'
##for i in glob(testpath+"/*"):
##    print(i)
##    filename =i
##    image_size=50
##    num_channels=3
##    images = []
##    # Reading the image using OpenCV
##    image = cv2.imread(filename)
##    #image=frame
##    cv2.imshow('test',image)
##    
##    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
##    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
##    images.append(image)
##    images = np.array(images, dtype=np.uint8)
##    images = images.astype('float32')
##    images = np.multiply(images, 1.0/255.0)
##
##    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
##    x_batch = images.reshape(1, image_size,image_size,num_channels)
##
##    ### Creating the feed_dict that is required to be fed to calculate y_pred 
##    feed_dict_testing = {x: x_batch, y_true: y_test_images}
##    result=sess.run(y_pred, feed_dict=feed_dict_testing)
##    # result is of this format [probabiliy_of_rose probability_of_sunflower]
##    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
##    print(result)
##    print(np.argmax(max(result)))
##    k = cv2.waitKey()
##    if k == 99:
##        continue
   
#Open Camera object
cap = cv2.VideoCapture(0)

#Decrease frame size (4=width,5=height)
cap.set(4, 700)
cap.set(5, 400)

h,s,v = 150,150,150
i=0
while(i<1000000):
    kernel = np.ones((5,5),np.uint8)

    ret, frame = cap.read()
        
    cv2.rectangle(frame, (300,300), (100,100), (0,255,0),0)
    crop_frame=frame[100:300,100:300]
     #Blur the image
    #blur = cv2.blur(crop_frame,(3,3))
    blur = cv2.GaussianBlur(crop_frame, (3,3), 0)
        
    #Convert to HSV color space
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    
    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
    med=cv2.medianBlur(mask2,5)
    med = cv2.morphologyEx(med, cv2.MORPH_OPEN, kernel)
    
    ##Displaying frames
    cv2.imshow('main',frame)
    cv2.imshow('masked',med)

    ##resizing the image
    med=cv2.resize(med,(50,50))
    ##Making it 3 channel
    med=np.stack((med,)*3)
    ##adjusting rows,columns as per x
    med=np.rollaxis(med,axis=1,start=0)
    med=np.rollaxis(med,axis=2,start=0)
    ##Rotating and flipping correctly as per training image
    M = cv2.getRotationMatrix2D((25,25),270,1)
    med = cv2.warpAffine(med,M,(50,50))
    med=np.fliplr(med)
    ##converting expo to float
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    ##printing index of max prob value
    ans=predict(med,y_test_images)
    #print(ans)
    #print(np.argmax(max(ans)))

    #Comparing for 50 continuous frames
    c_frame=np.argmax(max(ans))
    if(c_frame==p_frame):
        counter=counter+1
        p_frame=c_frame
        if (counter==thresholdframes):
            print(ans)
            print("Gesture:"+str(c_frame))
            if ( c_frame == 5 ):
                 PressKey(W)
                 time.sleep(1)
                 ReleaseKey(W)
#                PressKey(W)
#                time.sleep(2)
            elif (c_frame == 4):
                 PressKey(S)
                 time.sleep(1)
                 ReleaseKey(S)
#                PressKey(S)
#                time.sleep(1)
            elif (c_frame == 2):
                 PressKey(W)
                 PressKey(D)
                 time.sleep(0.6)
                 ReleaseKey(D)
                 ReleaseKey(W)
#                PressKey(W)
#                time.sleep(1)
#                PressKey(D)
#                time.sleep(.4)
            elif (c_frame==1 or c_frame==7 ):
                 PressKey(W)
                 PressKey(A)
                 time.sleep(0.6)
                 ReleaseKey(A)
                 ReleaseKey(W)
                
                
#                PressKey(W)
#                time.sleep(1)
#                PressKey(A)
#                time.sleep(.4)
            elif (c_frame == 8):
                old = current 
                current = 0x15
                releaseandswitch(old,current)
            else:
                ReleaseKey(current)
            
#            ReleaseKey(W)
#            ReleaseKey(S)
#            ReleaseKey(D)
#            ReleaseKey(A)
                 
            counter=0
            i=0
    else:
        p_frame=c_frame
        counter=0
        
 #close the output video by pressing 'ESC'
    k = cv2.waitKey(2) & 0xFF
    if k == 27:
        break
    i=i+1
    
cap.release()
cv2.destroyAllWindows()
