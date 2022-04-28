import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from agent import *

vectors=[(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)] #possible moves
bestGene=np.load("D:/Documents/Computer Science/Year 3/Dissertation/best.npy") #load file
whegBot=Agent_Conv2D((5*5)+2,[40,40],len(vectors)) #define the agent
whegBot.set_genes(bestGene) #setup agent
VectorBetween=[1,1]
#0,0,0 = 30cm
#35,35,35 = 60cm
#71,71,71 = 90cm
#107,107,107 = 120cm
#133,133,133 = 250cm
def point_obs(im,interval=200):
    width=im.shape[1]
    intervals=width//interval #get intervals of 200 pixels
    s=np.array([0 for i in range(intervals)])
    for i in range(intervals):
        s[i]=np.sum(im[:,i*interval:interval*(i+1)])
    best=np.argmax(s)
    val=np.sum(im[:,best*interval:interval*(best+1)])
    im[:,best*interval:interval*(best+1)]=255
    w, h = im[:,best*interval:interval*(best+1)].shape[:2]
    return im,(val)/(w*h)
def predict(image):
    action=whegBot.get_action(im,VectorBetween) 
    plt.title(action)
    plt.imshow(im)
    plt.show()
#open and save the demo image
#im=cv.imread("D:/Documents/Computer Science/Year 3/Dissertation/Results/Vision/Stereo2_alone.png")

#open up video capture
cap = cv.VideoCapture('C:/Users/Dexter Shepherd/Videos/2022-04-22-17-47-42.mp4')
if (cap.isOpened()== False):
  print("Error opening video stream or file")
  exit()
ret, im = cap.read()
im=np.dot(im, [0.2989, 0.5870, 0.1140]) #convert to greyscale
im=im[200:600] #limit to view
im=im[:,200:800]
im=cv.resize(im,(5,5))
#new,dat=point_obs(np.copy(im)) #get prediction
plt.imshow(im)
plt.cla()

#new=point_obs(im)
#get example
#plt.subplot(2,1,1)
#plt.title("Plot to show the view and predicted direction")
#plt.imshow(im)
#plt.subplot(2,1,2)
##plt.imshow(new)
#plt.savefig("D:/Documents/Computer Science/Year 3/Dissertation/Results/Vision/predicted idrection.png")
#plt.savefig("save.png")
#ad=cv.imread("save.png")
#get shape
#w, h = ad.shape[:2]
#print(w,h)
#fourcc = cv.VideoWriter_fourcc(*'mp4v')
#out = cv.VideoWriter('D:/Documents/Computer Science/Year 3/Dissertation/Results/Vision/output_pred.mp4', fourcc, 2, (w,h))
c=0
sums=[]
error=[]
while(cap.isOpened() and c<200):
    ret, im = cap.read()
    if ret and c%10==0:
        #print(c)
        im=np.dot(im, [0.2989, 0.5870, 0.1140]) #convert to greyscale
        im=im[200:600] #limit to view
        im=im[:,200:800]
        im=cv.resize(im,(5,5))
        #new,dat=point_obs(np.copy(im)) #get prediction
        predict(np.copy(im))
        w1, h1 = im.shape[:2]
        #calculations for debug
        sumOf=np.sum(im)
        #print("Amount of obstacle: ",1-(sumOf/(im.shape[0]*im.shape[1]*255))) #calculate messiness
        #sums.append(sumOf/(w1*h1))
        #error.append(dat)
        #display
        #if dat<300000:
        #    print("crash")
        #plt.subplot(2,1,1)
        #plt.title("Plot to show the view and predicted direction")
        #plt.imshow(im)
        #plt.subplot(2,1,2)
        #plt.title("Error rate "+str(dat))
        #plt.imshow(new)
        #plt.savefig("save.png")
        #ad=cv.imread("save.png")
        #cv.imshow('frame', ad)
        #if cv.waitKey(1) == ord('q'):
        #    break
        #ad=cv.resize(ad,(w,h)) 
        #out.write(ad)
    c+=1
cap.release()
#out.release()
plt.cla()
#plt.title("Changes in depth")
#plt.plot([i for i in range(len(sums))],sums,label="summed value of all pixels")
#plt.plot([i for i in range(len(sums))],error,label="Best direction summed pixels")
#plt.plot([i for i in range(len(sums))],[20 for i in range(len(error))],label="Threshold of obstacle",linestyle="--")
#plt.ylabel("Sum of pixels averaged")
#plt.xlabel("Frame in video")
#plt.legend(loc="upper right")
#plt.show()

