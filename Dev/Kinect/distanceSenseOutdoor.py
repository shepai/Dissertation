import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from agent import *

vectors=[(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)] #possible moves
bestGene=np.load("D:/Documents/Computer Science/Year 3/Dissertation/SavedModelsAndResults/best.npy") #load file
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
def prepIm(im,inter=cv.INTER_LINEAR):
    v_im=im.copy()
    origin=im[:,:1357].copy()
    disp=im[:,1358:].copy()
    disp1=np.dot(disp, [0.2989, 0.5870, 0.1140]) #convert to greyscale
    disp1=disp1[200:600] #limit to view
    disp1=disp1[:,200:800]
    im=cv.resize(disp1,(5,5),interpolation=inter) #resize to robot view
    return origin,disp,im
def GeneratePlot(origin,disp,im,action): #show the image
    #plt.figure(figsize=(3.5, 1.0))
    #fig, axes = plt.subplots(2,2, figsize=(3.5, 1.0))
    #plt.subplot(2,2,1)
    fig, axes = plt.subplots(1,4, figsize=(3.5, 2.0))
    axes[0].set_title("A", loc="left")
    axes[0].imshow(origin)
    axes[0].xaxis.set_visible(False)
    axes[0].yaxis.set_visible(False)
    #plt.subplot(2,2,2)
    axes[1].set_title("B", loc="left")
    axes[1].imshow(disp)
    axes[1].xaxis.set_visible(False)
    axes[1].yaxis.set_visible(False)
    #plt.subplot(2,2,3)
    axes[2].set_title("C", loc="left")
    axes[2].imshow(im)
    axes[2].xaxis.set_visible(False)
    axes[2].yaxis.set_visible(False)
    #plt.subplot(2,2,4)
    axes[3].set_title("D", loc="left")
    axes[3].arrow(0, 0, action[0], action[1], head_width=0.2, head_length=0.1, length_includes_head=True, facecolor="black")
    axes[3].set_xlim((-1.1, 1.1))
    axes[3].set_ylim((-1.1, 1.1))
    axes[3].set_aspect("equal")
    #axes[1,1].plot([0,action[0]],[0,action[1]])
    #axes[1,1].scatter(action[0],action[1],marker=">")
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    fig.tight_layout(pad=0.05)
    fig.savefig("D:/Documents/Computer Science/Year 3/Dissertation/Dev/Computer vision/OutSamples/"+"save"+".pdf")
    ad=cv.imread("D:/Documents/Computer Science/Year 3/Dissertation/Dev/Computer vision/OutSamples/"+"save"+".pdf")
    fig.savefig("D:/Documents/Computer Science/Year 3/Dissertation/Dev/Computer vision/OutSamples/"+"save"+".png")
    ad=cv.imread("D:/Documents/Computer Science/Year 3/Dissertation/Dev/Computer vision/OutSamples/"+"save"+".png")
    plt.show()
    #plt.clf()
    return ad
#open up video capture
cap = cv.VideoCapture('C:/Users/Dexter Shepherd/Videos/2022-04-30-10-00-24.mp4')
if (cap.isOpened()== False):
  print("Error opening video stream or file")
  exit()
ret, im = cap.read()
origin,disp,im0=prepIm(im)

im=im0.copy()
plt.clf()
p=GeneratePlot(origin,disp,im,[1,1])
"""
p=GeneratePlot(origin,disp,im,[1,0])
p=GeneratePlot(origin,disp,im,[-1,-1])
p=GeneratePlot(origin,disp,im,[-1,0])
"""

#plt.savefig("D:/Documents/Computer Science/Year 3/Dissertation/Results/Vision/predicted idrection.png")
#plt.savefig("save.png")

h, w = p.shape[:2]
print(w,h)
fourcc = cv.VideoWriter_fourcc(*'mp4v')
#out = cv.VideoWriter('D:/Documents/Computer Science/Year 3/Dissertation/Results/Vision/output_terrain_area.mp4', fourcc, 10, (w,h))
c=0
sums=[]
error=[]
cap.set(cv.CAP_PROP_POS_MSEC, 6000)
while(cap.isOpened() and c<300):
    ret, im = cap.read()
    if ret and c%1==0:
        #print(c)
        origin,disp,im=prepIm(im,inter=cv.INTER_AREA)
        act=predict(np.copy(im))
        p=GeneratePlot(origin,disp,im,act)
        w1, h1 = im.shape[:2]
        #calculations for debug
        #cv.imshow('frame', p)
        #if cv.waitKey(1) == ord('q'):
        #    break
        #p=cv.resize(p,(w,h)) 
        #out.write(p)
    c+=1
cap.release()
#out.release()
plt.cla()


"""
for i in range(200):
    ret, im = cap.read()
    if i%10==0:
        origin,disp,im0=prepIm(im)
        origin,disp,im1=prepIm(im,inter=cv.INTER_CUBIC)
        origin,disp,im2=prepIm(im,inter=cv.INTER_AREA)
        origin,disp,im3=prepIm(im,inter=cv.INTER_LANCZOS4)

        plt.subplot(4,2,1)
        plt.title("Original")
        plt.imshow(origin)
        plt.subplot(4,2,2)
        plt.title("Stereo")
        plt.imshow(disp)
        plt.subplot(4,2,3)
        plt.title("Linear")
        plt.imshow(im0)
        plt.subplot(4,2,4)
        plt.title("Cubic")
        plt.imshow(im1)
        plt.subplot(4,2,5)
        plt.title("Area")
        plt.imshow(im2)
        plt.subplot(4,2,6)
        plt.title("LANCZOS4")
        plt.imshow(im3)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.7)
        plt.show()

"""