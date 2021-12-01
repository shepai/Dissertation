#noise generator for simulation plot
import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random as rnd
import copy
from agent import Agent

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

def mutation(gene, mean=0, std=0.5):
    gene = gene + np.random.normal(mean, std, size=gene.shape) #mutate the gene via normal 
    # constraint
    gene[gene >4] = 4
    gene[gene < -4] = -4
    return gene

def crossover(loser, winner, p_crossover=0.5): #provide a crossover function
    for i,gene in enumerate(winner):
      if random.random() <= p_crossover:
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

def fitness(broke,energy,mx,route):
    if broke:
        return 0
    print(sum(route),mx)
    return 1-(max(0,energy)/(mx+sum(route)))

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

def readIm(terrain,point,r=5):
    #read the ground around the agent at a radius of i
    
    x = np.arange(0, len(terrain))
    y = np.arange(0, len(terrain))
    arr=copy.deepcopy(terrain)
    cx = point[1]
    cy = point[0]
    # The two lines below could be merged, but I stored the mask
    # for code clarity.
    mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
    return np.array(arr[mask])
    
    layers=[]
    #for m in 
    LOffset,ROffset=0,0
    return layers
    

def microbial_trial(genes):
    return genes
world,shape=generateWorld()
startPos=[int(SIZE/2),int(SIZE/2)] #centre point

testIm=readIm(world,startPos)#.flatten()
print(testIm)
Generations=50
vectors=[(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)] #possible moves

whegBot=Agent(testIm.shape[0],5,5,5,len(vectors)) #define the agent

pop_size=10
gene_pop=[]
for i in range(pop_size): #vary from 10 to 20 depending on purpose of robot
    gene=np.random.normal(0, 0.8, (whegBot.num_genes))
    gene_pop.append(copy.deepcopy(gene))#create

  
for gen in range(Generations):
    #generate the world terrain
    world,shape=generateWorld()
    world=np.pad(np.array(world), (2,2), 'constant',constant_values=(-6,-6))
    world=np.pad(np.array(world), (3,3), 'constant',constant_values=(-7,-7))
    world=np.pad(np.array(world), (1,1), 'constant',constant_values=(-8,-8))
    #randomly pick a start position
    startPos=pickPosition(world,4,LBounds=6)
   
    maxPath=30
    pathx=[]
    pathy=[]
    current=startPos.copy()
    energy=0
    last=startPos.copy()
    im=readIm(world,startPos)
    broke=False
    routeValues=[]
    assert len(im)==69, "Panoramic Camera failed"+str(len(im)) #assert length correct
    for i in range(maxPath): #loop through and generate path
        #print(readIm(world,))
        v=rnd.choice(vectors)
        pathx.append(current[0]+v[0])
        pathy.append(current[1]+v[1])
        last=current.copy()
        current[0]+=v[0]
        current[1]+=v[1]
        if current[0]>=0 and current[0]<len(world)-1 and current[1]>=0 and current[1]<len(world[0])-1:
            if world[current[0]][current[1]]<=-6:
                print("water")
                broke=True
                break
            else:
                climb=max(-1,world[current[0]][current[1]]-world[last[0]][last[1]]) #non 0 value of total climb
                routeValues.append(abs(climb))
                energy+=1+climb
    print("total energy consumed",energy,"fitness",fitness(broke,energy,maxPath,routeValues))
    plt.plot(pathy,pathx)
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

