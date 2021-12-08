"""
Noise generator using perlin noise
Generate depth perception images from any coordinate 

Code by Dexter Shepherd
"""
import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random as rnd
import copy
from matplotlib.collections import EllipseCollection
import math as maths
import os
import time

SIZE=50
def generateWorld():
    shape = (SIZE,SIZE)
    scale = 100.0
    octaves = rnd.randint(2,20)
    persistence = 0.5
    lacunarity = 2

    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(i/scale, 
                                        j/scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=1024, 
                                        repeaty=1024, 
                                        base=42)
    world=world*100 #normalize numbers
    world=world.astype(int)
    print("Octaves:",octaves)
    return world,shape



def pickPosition(terrain,value,deny=[],LBounds=8,UBounds=4):
    current=-100
    cords=[0,0]
    deny.append(cords)
    while cords in deny:
        while current<value-LBounds or current>value+UBounds:
            cords=[rnd.randint(0,SIZE-1),rnd.randint(0,SIZE-1)]
            current=terrain[cords[0],cords[1]]
    return cords


#canReach will make sure the problem is solvable
def canReach(terrain,start,goal,endmarked=[[False for i in range(SIZE)] for j in range(SIZE)]):
    #check whether the bot can reach the other
    #expand out from end and make paths
    y,x=start[0],start[1]
    val=False
    if terrain[x][y]!=-1 and not endmarked[x][y]:
        endmarked[x][y]=True
        if x-1>=0:
            val=canReach(terrain,[x-1,y],goal,endmarked=endmarked)
        if x+1<SIZE:
            val=canReach(terrain,[x+1,y],goal,endmarked=endmarked)
        if y-1>0:
            val=canReach(terrain,[x,y-1],goal,endmarked=endmarked)
        if y+1<SIZE:
            val=canReach(terrain,[x,y+1],goal,endmarked=endmarked)
    if x==goal[1] and y==goal[0]:
        val=True
    return val
    
    

#getBestRoute will be used to measure how fit an evolved route is
def getBestRoute(terrain,start,end):
    #find the least cost route from A to B
    #return metrics
    
    return []
def expand(terrain,Map,start):
    for i in range(len(terrain)):
        for j in range(len(terrain[i])):
            if terrain[i][j]<-5:
                Map[i][j]=-1
            elif [j,i]==start:
                Map[i][j]=0
            else:
                Map[i][j]=terrain[i][j]+getDist([j,i],start)
    return Map

def getDist(start,end):
    d1=((start[0]-end[0])**2 + (start[1]-end[1])**2)**0.5
    return int(d1)

def readIm(map,position,direction,d=5):
    #read the ground around the agent at a radius of i
    r=maths.radians(direction)
    vector=(int(d*maths.cos(r)),int(d*maths.sin(r)))
    x,y=position
    print(x,y)
    line1=[[x+round(abs(d-i)*maths.cos(r)),y+round(abs(d-i)*maths.sin(r))] for i in range(5)]
    line2=[[x+round(abs(d-i)*maths.cos(r+maths.radians(10))),y+round(abs(d-i)*maths.sin(r+maths.radians(10)))] for i in range(5)]
    line3=[[x+round(abs(d-i)*maths.cos(r+maths.radians(20))),y+round(abs(d-i)*maths.sin(r+maths.radians(20)))] for i in range(5)]
    line4=[[x+round(abs(d-i)*maths.cos(r+maths.radians(30))),y+round(abs(d-i)*maths.sin(r+maths.radians(30)))] for i in range(5)]
    for i in range(len(map)):
        for j in range(len(map[i])):
            if [j,i] in line1 or [j,i] in line2 or [j,i] in line3 or [j,i] in line4:
                print(1,end = "")
            elif position==[j,i]:
                print("O",end = "")
            else:
                print(0,end = "")
        print("")

    plt.plot(line1)
    plt.plot(line2)
    plt.plot(line3)
    plt.plot(line4)
    return []

def build3D(world):
    m=np.array([[[0 for i in range(np.amax(world)+abs(np.amin(world)))] for j in range(len(world[i]))]for i in range(len(world))])
    for i in range(len(world)):
        for j in range(len(world[i])):
            for z in range(abs(np.amin(world))+max(world[i][j],0)):
                m[i][j][z]=1
    return m
i=0
while True:
    #generate the world terrain
    world,shape=generateWorld() 
    #randomly pick a start position
    startPos=pickPosition(world,4,LBounds=6)
    vectors=[(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)] #possible moves
    map=build3D(world) 
    print("Angle",i)
    im=readIm(map,startPos,i)
    i+=10
    #time.sleep(1)
    #os.system('cls')
    #print(canReach(Rmap,startPos,endPos))
    im = plt.imshow(world,cmap='terrain')
    cb = plt.colorbar(im)
    plt.setp(cb.ax.get_yticklabels([-1,0,1]), visible=False)
    plt.scatter(startPos[0],startPos[1])
    plt.show()
    """
    lin_x = np.linspace(0,1,shape[0],endpoint=False)
    lin_y = np.linspace(0,1,shape[1],endpoint=False)
    x,y = np.meshgrid(lin_x,lin_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x,y,world,cmap='terrain')


    plt.show()
    #"""

"""
    maxPath=30
    pathx=[]
    pathy=[]
    current=startPos.copy()
    energy=0
    last=startPos.copy()
    for i in range(maxPath):
        v=rnd.choice(vectors)
        pathx.append(current[0]+v[0])
        pathy.append(current[1]+v[1])
        last=current.copy()
        current[0]+=v[0]
        current[1]+=v[1]
        if current[0]>=0 and current[0]<len(world) and current[1]>=0 and current[1]<len(world[0]):
            if world[current[0]][current[1]]<=-6:
                print("water")
            else:
                climb=max(0,world[current[0]][current[1]]-world[last[0]][last[1]]) #non 0 value of total climb
                energy+=1+climb
                
    print("total energy consumed",energy)
    plt.plot(pathy,pathx)
    #"""
