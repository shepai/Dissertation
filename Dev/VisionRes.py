import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import copy
from skimage.metrics import structural_similarity


DATAPATH = "D:\Documents\Computer Science\Year 3\Dissertation\Dev\Computer vision\StereoDat\ "
def readAndStereo(imgOriginalSceneA,imgOriginalSceneB):
    #stereo creation
    frame0_new=cv.cvtColor(imgOriginalSceneA, cv.COLOR_BGR2GRAY)
    frame1_new=cv.cvtColor(imgOriginalSceneB, cv.COLOR_BGR2GRAY)
    
    #blurr
    frame0_new = cv.filter2D(frame0_new,-1,kernel)
    frame1_new = cv.filter2D(frame1_new,-1,kernel)
    
    #for i in range(16,int(16*5),16): #display different maps
    stereo = cv.StereoBM_create(numDisparities=16*2, blockSize=27) #gen stereo
        
    disparity = stereo.compute(frame0_new,frame1_new) #get map

    norm = np.zeros(frame0_new.shape) #create empty frame
    disp = cv.normalize(disparity, norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U) #normalize
    return disp,frame0_new,frame1_new
def Kmeans(frame0_new,frame1_new):
    #convert to float
    Z1 = frame0_new.reshape((-1,3))
    Z2 = frame1_new.reshape((-1,3))

    frame0_new_fl = np.float32(Z1)
    frame1_new_fl = np.float32(Z2)
    #reduce and place back
    
    ret,label,center=cv.kmeans(frame0_new_fl,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    frame0_new = res.reshape((frame0_new.shape))
    ret,label,center=cv.kmeans(frame0_new_fl,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    frame1_new = res.reshape((frame1_new.shape))
    return frame0_new,frame1_new
def getSame(before_gray,after_gray):
    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV)[1]
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before_gray.shape, dtype='uint8')
    filled_after = after_gray.copy()

    for c in contours:
        area = cv.contourArea(c)
        if area > 40:
            x,y,w,h = cv.boundingRect(c)
            cv.drawContours(mask, [c], 0, (0), -1)
            cv.drawContours(filled_after, [c], 0, (0), -1)

    return filled_after

capA = cv.VideoCapture(2)
capB = cv.VideoCapture(3)

not_broken=True

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
kernel = np.ones((5,5),np.float32)/25

imgOriginalSceneA = cv.imread("D:\Documents\Computer Science\Year 3\Dissertation\Dev\Computer vision\StereoDat\ 3L.png")
imgOriginalSceneB = cv.imread("D:\Documents\Computer Science\Year 3\Dissertation\Dev\Computer vision\StereoDat\ 3R.png")
Pastdisp,frame0_new,frame1_new=readAndStereo(imgOriginalSceneA,imgOriginalSceneB)
c=0
disp,frame0_new,frame1_new=readAndStereo(imgOriginalSceneB,imgOriginalSceneA)
#disp=getSame(Pastdisp,disp) #remove differences
#disp = cv.dilate(disp,kernel,iterations = 3)

plt.subplot(1,3,1)
plt.imshow(imgOriginalSceneB)
plt.subplot(1,3,2)
plt.imshow(imgOriginalSceneA)
plt.subplot(1,3,3)
plt.imshow(disp)
plt.title("Results of stereo imaging on the Arducam")

plt.show()