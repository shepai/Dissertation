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
kernel = np.ones((5,5),np.float32)/25
#0,0,0 = 30cm
#35,35,35 = 60cm
#71,71,71 = 90cm
#107,107,107 = 120cm
#133,133,133 = 250cm
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
def predict(image):
    action=whegBot.get_action(im,VectorBetween) 
    return action
#open and save the demo image
#im=cv.imread("D:/Documents/Computer Science/Year 3/Dissertation/Results/Vision/Stereo2_alone.png")
def prepIm(imA,imB,inter=cv.INTER_AREA):
    imB=imB[40:].copy()
    imA=imA[:len(imA)-40].copy()
    disp,frame0_new,frame1_new=readAndStereo(imB,imA)
    #disp1=np.dot(disp, [0.2989, 0.5870, 0.1140]) #convert to greyscale
    disp1=disp[200:600] #limit to view
    disp1=disp[:,200:600]
    
    im=cv.resize(disp1,(5,5),interpolation=inter) #resize to robot view
    return imB,disp,im
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
    plt.plot([0,action[0]],[0,action[1]])
    plt.scatter(action[0],action[1],marker=">")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.savefig("D:/Documents/Computer Science/Year 3/Dissertation/Dev/Computer vision/OutSamples/"+"save"+".png")
    ad=cv.imread("D:/Documents/Computer Science/Year 3/Dissertation/Dev/Computer vision/OutSamples/"+"save"+".png")
    plt.clf()
    return ad

#open up video capture
#cap = cv.VideoCapture('C:/Users/Dexter Shepherd/Videos/2022-04-30-10-00-24.mp4')

im1 = cv.imread("D:\Documents\Computer Science\Year 3\Dissertation\Dev\Computer vision\StereoDat\ 2L.png")
im2 = cv.imread("D:\Documents\Computer Science\Year 3\Dissertation\Dev\Computer vision\StereoDat\ 2R.png")



origin,disp,im0=prepIm(im1,im2)
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
out = cv.VideoWriter('D:/Documents/Computer Science/Year 3/Dissertation/Results/Vision/output_terrain_area_arducam.mp4', fourcc, 5, (w,h))
c=1
sums=[]
error=[]
while(c<24):
    imA = cv.imread("D:\Documents\Computer Science\Year 3\Dissertation\Dev\Computer vision\StereoDat\ "+str(c)+"L.png")
    imB = cv.imread("D:\Documents\Computer Science\Year 3\Dissertation\Dev\Computer vision\StereoDat\ "+str(c)+"R.png")
    if c%1==0:
        #print(c)
        origin,disp,im=prepIm(imA,imB,inter=cv.INTER_AREA)
        act=predict(np.copy(im))
        p=GeneratePlot(origin,disp,im,act)
        w1, h1 = im.shape[:2]
        #calculations for debug
        cv.imshow('frame', p)
        if cv.waitKey(1) == ord('q'):
            break
        #p=cv.resize(p,(w,h)) 
        out.write(p)
    c+=1

out.release()
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