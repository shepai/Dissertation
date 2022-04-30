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

def predict(image):
    action=whegBot.get_action(im,VectorBetween) 
    return action
#open and save the demo image
#im=cv.imread("D:/Documents/Computer Science/Year 3/Dissertation/Results/Vision/Stereo2_alone.png")
def prepIm(im):
    v_im=im.copy()
    origin=im[:,:1357].copy()
    disp=im[:,1358:].copy()
    disp1=np.dot(disp, [0.2989, 0.5870, 0.1140]) #convert to greyscale
    disp1=disp1[200:600] #limit to view
    disp1=disp1[:,200:800]
    im=cv.resize(disp1,(5,5)) #resize to robot view
    return origin,disp,im

def GeneratePlot(origin,disp,im,action): #show the image
    plt.subplot(2,2,1)
    plt.title("Original image")
    plt.imshow(origin)
    plt.subplot(2,2,2)
    plt.title("Depth image")
    plt.imshow(disp)
    plt.subplot(2,2,3)
    plt.title("Generated robot vision")
    plt.imshow(im)
    plt.subplot(2,2,4)
    plt.title("Vector prediction: "+str(action))
    plt.plot([0,action[1]],[0,action[0]])
    plt.scatter(action[1],action[0],marker=">")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.show()
    plt.savefig("save.png")
    ad=cv.imread("save.png")
    plt.clf()
    return ad
#open up video capture
cap = cv.VideoCapture('C:/Users/Dexter Shepherd/Videos/2022-04-30-10-00-24.mp4')
if (cap.isOpened()== False):
  print("Error opening video stream or file")
  exit()
ret, im = cap.read()
origin,disp,im=prepIm(im)
p=GeneratePlot(origin,disp,im,[1,1])
p=GeneratePlot(origin,disp,im,[1,0])
p=GeneratePlot(origin,disp,im,[-1,-1])
p=GeneratePlot(origin,disp,im,[-1,0])
exit()

#plt.savefig("D:/Documents/Computer Science/Year 3/Dissertation/Results/Vision/predicted idrection.png")
#plt.savefig("save.png")

w, h = p.shape[:2]
print(w,h)
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('D:/Documents/Computer Science/Year 3/Dissertation/Results/Vision/output_terrain.mp4', fourcc, 2, (w,h))
c=0
sums=[]
error=[]
while(cap.isOpened() and c<200):
    ret, im = cap.read()
    if ret and c%10==0:
        #print(c)
        origin,disp,im=prepIm(im)
        #new,dat=point_obs(np.copy(im)) #get prediction
        act=predict(np.copy(im))
        p=GeneratePlot(origin,disp,im,act)
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

