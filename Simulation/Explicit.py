#noise generator for simulation plot
import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random as rnd
import math as maths

SIZE=50
def generateWorld():
    shape = (SIZE,SIZE)
    scale = 100.0
    octaves = 10 #rnd.randint(2,20)
    persistence = 0.7
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
    #print("Octaves:",octaves)
    return world,shape

def mutation(gene, mean=0, std=0.5):
    gene = gene + np.random.normal(mean, std, size=gene.shape) #mutate the gene via normal 
    # constraint
    gene[gene >4] = 4
    gene[gene < -4] = -4
    return gene

def crossover(loser, winner, p_crossover=0.5): #provide a crossover function
    for i,gene in enumerate(winner):
      if rnd.random() <= p_crossover:
        loser[i] = winner[i]
    return loser

def pickPosition(terrain,value,deny=[],LBounds=8,UBounds=4):
    current=-100
    cords=[0,0]
    deny.append(cords)
    while cords in deny:
        while current<value-LBounds or current>value+UBounds:
            cords=[rnd.randint(0,SIZE-1),rnd.randint(0,SIZE-1)]
            current=terrain[cords[0],cords[1]]
    return cords

def fitness(broke,energy,endDist):
    if broke or endDist>10:
        return 0
    #print(sum(route),mx)
    #return 1-(max(0,energy)/(mx+sum(route)))
    return (((100-energy)/100)*0.3 + ((10-endDist)/10)*0.7)*100

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

def getSlice(map,line,position,imH=5):
    #move up a height
    height=map[position[0]][position[1]]
    #max(0,height-map[coord[0]][coord[1]])
    column=[]
    past=0
    c=[]
    passes=0 #allow building
    
    for coord in line[::-1]:
        column.append(map[coord[0]][coord[1]]-height+50)

    return column
def readIm(map,position,direction,imSize=(5,5),d=5):
    #read the ground around the agent at a radius of i
    assert (imSize[1]*20<360) #if the image size requires larger than panoramic this will be stupid
    r=maths.radians(direction)
    vector=(int(d*maths.cos(r)),int(d*maths.sin(r)))
    x,y=position
    lines=[]
    ANG=0
    for pixX in range(imSize[1]):
        lines.append(
            np.array([[x+round(abs(d-i)*maths.cos(r+maths.radians(ANG))),y+round(abs(d-i)*maths.sin(r+maths.radians(ANG)))] for i in range(imSize[0])])
        )
        ANG+=20 #increase angle each time
    ##check through each line
    #place pixel in the height relevant based on terrain height
    A=[]
    for lin in lines:
        A.append(getSlice(map,lin,position,imH=imSize[0]))

    A=np.array(A)/50
    A = A.flatten()
    return A

def getCircleCoord(centre,radius):
    #(x-centre[0])^2 + (y-centre[1])^2 = radius^2
    coords=[]
    bx=-2*(centre[0])
    cx=centre[0]**2
    by=-2*(centre[1])
    cy=centre[1]**2
    for y in range(centre[1]-radius,centre[1]+radius,1):
        c=(y**2) + by*(y) + cy #get y filled in
        c=cx+c-(radius**2) #normalize for x quadratic
        #apply quadratic form
        x1=((-1*bx)+maths.sqrt((bx**2)-(4*c)))/(2)
        x2=((-1*bx)-maths.sqrt((bx**2)-(4*1*c)))/(2)
        coords.append([x1,y])
        coords.append([x2,y])
    return coords

world,shape=generateWorld()
startPos=[int(SIZE/2),int(SIZE/2)] #centre point

Generations=200
vectors=[(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)] #possible moves

fitnesses=[]
for gen in range(Generations):
    print("Gen",gen+1)
    #generate the world terrain
    world,shape=generateWorld()
    world=np.pad(np.array(world), (2,2), 'constant',constant_values=(-6,-6))
    world=np.pad(np.array(world), (3,3), 'constant',constant_values=(-7,-7))
    world=np.pad(np.array(world), (1,1), 'constant',constant_values=(-8,-8))
    #randomly pick a start position
    