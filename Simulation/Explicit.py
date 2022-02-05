#noise generator for simulation plot
import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random as rnd
import math as maths
import random
import copy

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

def getDirection(cords,current,image): #get the nearest vector
    #will need to:
    #weight the best direction to take via vector
    #weight the best direction based on image
    #combine weights and pick vector
    VectorBetween=[cords[0]-current[0],cords[1]-current[1]]
    vectorMag=[]
    mags={}
    #pointAim=[current[0]+VectorBetween[0],current[1]+VectorBetween[1]] #get point aiming for
    nearest=[0,0]
    nearestDist=100
    currentH=world[current[1]][current[0]]

    for vec in vectors:
        point=[current[0]+vec[0],current[1]+vec[1]] #apply vector to current coord
        x=getDist(point,cords)
        h=world[point[1]][point[0]]
        if nearestDist>x and abs(currentH-h)<5 and h>-6:
            nearestDist=x
            nearest=copy.deepcopy(vec)

    return nearest

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
    startPos=[int(SIZE/2),int(SIZE/2)] #centre point
    valid=False
    cords=[]
    radius=10
    coords=getCircleCoord(startPos,radius)
    while not valid: #validate random destination
        cords=random.choice(coords)
        tmp=cords
        cords=[int(cords[0]),int(cords[1])]
        if world[cords[1]][cords[0]]>-6 and world[cords[1]][cords[0]]<=10:
            valid=True
        else:
            coords.remove(tmp) #prevent from picking
    cords=np.array(cords)
    i=0
    last=0
    broke=False
    current=copy.deepcopy(startPos)
    runs=30
    pathx=[]
    pathy=[]
    energy=0
    v=rnd.choice(vectors)
    
    while i<runs and not broke and getDist(current,cords)>1: #loop through and generate path
        dir=maths.cos(v[1]) #get angle from y-axis
        im=readIm(world,current,dir) #read the image that the agent sees
        last=copy.deepcopy(current)
        v=getDirection(cords,current,im) #get chosen direction
        current[0]+=v[0]
        current[1]+=v[1]
        pathx.append(current[0])
        pathy.append(current[1])
        lastH=world[current[1]][current[0]]
        if current[0]>=0 and current[0]<len(world[0])-1 and current[1]>=0 and current[1]<len(world)-1:
            if world[current[1]][current[0]]<=-6 or lastH-world[current[1]][current[0]]>3: #do not allow the rover to enter water
                #or if there is a substanial step
                broke=True
            else: #calculate energy usage
                climb=max(-1,world[current[1]][current[0]]-world[last[1]][last[0]]) #non 0 value of total climb
                energy+=1+climb
        i+=1
        
    fit=fitness(broke,energy,getDist(current,cords))
    fitnesses.append(fit)
    """
    plt.plot(pathx,pathy) #show best path
    plt.title("Gene "+str(fit)+"% after generations")
    plt.scatter(cords[0],cords[1])
    plt.scatter(startPos[0],startPos[1],c="r")
    #print(canReach(Rmap,startPos,endPos))
    plt.imshow(world,cmap='terrain') #show best show
    plt.show()
    #"""


fitnesses=np.around(np.array(fitnesses),1)
d={}
for i in fitnesses:
    d[i]=d.get(i,0)+1

dictionary_items = d.items()
d=sorted(dictionary_items)
print(d)
d=dict(d)
plt.bar(range(len(d)), list(d.values()), align='center')
plt.xticks(range(len(d)), list(d.keys()))
# # for python 2.x:
# plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
# plt.xticks(range(len(D)), D.keys())  # in python 2.x
plt.title("Bar graph of rounded fitnesses to 1dp over "+str(Generations)+" generations")

plt.ylabel("Amount of trials")
plt.xlabel("Fitness Units")
plt.show()


plt.plot([i for i in range(len(fitnesses))],fitnesses)
plt.title("Results of population fitness over "+str(Generations)+" generations")
plt.ylabel("Fitness Units")
plt.xlabel("Generation")
plt.legend(loc="lower right")
plt.show()
