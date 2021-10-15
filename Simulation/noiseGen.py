#noise generator for simulation plot
import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random as rnd
import copy

SIZE=25
def generateWorld():
    shape = (SIZE,SIZE)
    scale = 100.0
    octaves = 4
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

def expand__old(terrain,Map,position,start=[0,0]):
    y,x=position[0],position[1]
    if Map[x][y]==0:
        if terrain[x][y]<-5: #Mark out water and obstacles
            Map[x][y]=-1
        else:
            Map[x][y]=getDist([x,y],start) + terrain[x][y] #add weight
        #check all directions recursevly
        if x-1>=0:
            Map=expand(terrain,copy.deepcopy(Map),[x-1,y],start=start)
        if x+1<SIZE:
            Map=expand(terrain,copy.deepcopy(Map),[x+1,y],start=start)
        if y-1>0:
            Map=expand(terrain,copy.deepcopy(Map),[x,y-1],start=start)
        if y+1<SIZE:
            Map=expand(terrain,copy.deepcopy(Map),[x,y+1],start=start)
    return copy.deepcopy(Map) #return new map

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

while True:
    #generate the world terrain
    world,shape=generateWorld() 
    #randomly pick a start position
    startPos=pickPosition(world,4)
    #randomly pick an end position
    endPos=pickPosition(world,8,deny=copy.deepcopy(startPos),UBounds=15)
    plt.scatter(startPos[1],startPos[0],c="b")
    plt.scatter(endPos[1],endPos[0],c="r")
    Rmap=[[0 for i in range(SIZE)] for j in range(SIZE)]
    m=expand(world,copy.deepcopy(Rmap),startPos)
    
    for row in m:
        print(row)
    #print(canReach(Rmap,startPos,endPos))
    plt.imshow(world,cmap='terrain')
    plt.show()
    """
    lin_x = np.linspace(0,1,shape[0],endpoint=False)
    lin_y = np.linspace(0,1,shape[1],endpoint=False)
    x,y = np.meshgrid(lin_x,lin_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x,y,world,cmap='terrain')

    ax.scatter((startPos[0]+1)/SIZE,(startPos[1]+1)/100,world[startPos[0]][startPos[1]]+2,c="b")
    ax.scatter((endPos[0]+1)/SIZE,(endPos[1]+1)/100,world[endPos[0]][endPos[1]]+2,c="r")

    plt.show()
    #"""

