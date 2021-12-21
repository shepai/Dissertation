#noise generator for simulation plot
import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random as rnd
import copy
from agent import *
import math as maths

SIZE=50
def generateWorld():
    shape = (SIZE,SIZE)
    scale = 100.0
    octaves = 20 #rnd.randint(2,20)
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

def getSlice(map,line,position,WATER_LEVEL=4):
    #move up a height
    height=np.count_nonzero(map[position[0]][position[1]]==1)
    #max(0,height-map[coord[0]][coord[1]])
    column=[]
    past=0
    c=[]
    
    for coord in line[::-1]: #loop through coordinates
        coord=coord[::-1]
        try:
            c.append(np.count_nonzero(map[coord[0]][coord[1]] == 1))
            count = max(np.count_nonzero(map[coord[0]][coord[1]] == 1)-height,0)
            #look at depth of block in front of
            if np.count_nonzero(map[coord[0]][coord[1]] == 1)<=WATER_LEVEL: #show dangers
                column.append(-1)
            else:
                column.append(max(count,past))
            if np.count_nonzero(map[coord[0]][coord[1]]== 1)<height: #if still lower than current
                past=count
            else: past=max(past,count)
        except IndexError: #if line outside of map bounds
            column.append(0)
            c.append(0)
    return column
def readIm(map,position,direction,imSize=(5,5),d=5):
    #read the ground around the agent at a radius of i
    assert (imSize[1]*20<360) #if the image size requires larger than panoramic this will be stupid
    r=direction
    vector=(int(d*maths.cos(r)),int(d*maths.sin(r)))
    x,y=position
    image=[]
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
        A.append(getSlice(map,lin,position))
    A=np.array(A)*10
    A = A.flatten()
    return A

def build3D(world):
    #build a 3d representation
    zSize=abs(np.amin(world))+abs(np.amax(world))    
    m=np.array([[[0 for i in range(np.amax(world)+abs(np.amin(world)))] for j in range(len(world[i]))]for i in range(len(world))])
    for i in range(len(world)):
        for j in range(len(world[i])):
            position=world[i][j]
            bound=0 #set the bound of size
            if position<0: bound=10-abs(position)
            else: bound=10+position
            for z in range(zSize): #show redundant space
                if z<=bound: #only ad ones to phase
                    m[i][j][z]=1
            #
    return m

def run_trial(gene,runs=30):
    pathx=[]
    pathy=[]
    current=startPos.copy()
    energy=0
    last=startPos.copy()
    broke=False
    routeValues=[]
    v=rnd.choice(vectors)
    whegBot.set_genes(gene) #set the genes of the agent
    map=build3D(world) 
    for i in range(runs): #loop through and generate path
        dir=maths.cos(v[1]) #get angle from y-axis
        im=readIm(map,current,dir) #read the image that the agent sees
        assert len(im)==25, "Panoramic Camera failed"+str(len(im)) #assert length correct
        v=whegBot.get_action(im) #get action from the agent
        last=current.copy()
        pathx.append(current[0]+v[0])
        pathy.append(current[1]+v[1])
        current[0]+=v[0]
        current[1]+=v[1]
        if current[0]>=0 and current[0]<len(world)-1 and current[1]>=0 and current[1]<len(world[0])-1:
            if world[current[0]][current[1]]<=-6: #do not allow the rover to enter water
                print("water")
                broke=True
                break
            else: #calculate energy usage
                climb=max(-1,world[current[0]][current[1]]-world[last[0]][last[1]]) #non 0 value of total climb
                routeValues.append(abs(climb))
                energy+=1+climb
    print("total energy consumed",energy,"fitness",fitness(broke,energy,runs,routeValues))
    return pathy,pathx,fitness(broke,energy,runs,routeValues)

def microbial(genes,world,position):
    global BESTFIT
    global BEST
    #microbial algorithm trial
    ind_1 = rnd.randint(0,len(genes)-1)
    ind_2 = rnd.randint(0,len(genes)-1)
    #get two random positions
    gene1=mutation(genes[ind_1])
    gene2=mutation(genes[ind_2])
    #run trial for each
    p1x,p1y,fitness1=run_trial(gene1)
    p2x,p2y,fitness2=run_trial(gene2)
    #show results
     
    #microbial selection
    if fitness1>fitness2:
        gene2=copy.deepcopy(crossover(gene2,gene1))
    elif fitness2>fitness1:
        gene1=copy.deepcopy(crossover(gene1,gene1))
    if max(fitness1,fitness2)>BESTFIT:
        BESTFIT=max(fitness1,fitness2) #gather the maximum
        if fitness1>fitness2: BEST=[p1x,p1y,copy.deepcopy(world)]
        else: BEST=[p2x,p2y,copy.deepcopy(world)]
    genes[ind_1]=copy.deepcopy(gene1)
    genes[ind_2]=copy.deepcopy(gene2)
    return genes,max(fitness1,fitness2)
BEST=[]
BESTFIT=0
world,shape=generateWorld()
startPos=[int(SIZE/2),int(SIZE/2)] #centre point

map=build3D(world)
testIm=readIm(map,[25,25],30) #read the image that the agent sees
Generations=500
vectors=[(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)] #possible moves

whegBot=Agent_defineLayers(testIm.shape[0],[10,10],len(vectors)) #define the agent

pop_size=10
gene_pop=[]
for i in range(pop_size): #vary from 10 to 20 depending on purpose of robot
    gene=np.random.normal(0, 0.8, (whegBot.num_genes))
    gene_pop.append(copy.deepcopy(gene))#create

fitnesses=[]
for gen in range(Generations):
    print("Gen",gen+1)
    #generate the world terrain
    world,shape=generateWorld()
    world=np.pad(np.array(world), (2,2), 'constant',constant_values=(-6,-6))
    world=np.pad(np.array(world), (3,3), 'constant',constant_values=(-7,-7))
    world=np.pad(np.array(world), (1,1), 'constant',constant_values=(-8,-8))
    #randomly pick a start position
    startPos=pickPosition(world,4,LBounds=6)
    #genes have been selected
    gene_pop,fit=microbial(gene_pop,world,startPos)
    fitnesses.append(max([fit]+fitnesses))


plt.plot(BEST[1],BEST[0]) #show best path
#print(canReach(Rmap,startPos,endPos))
plt.imshow(BEST[2],cmap='terrain') #show best show
plt.show()

plt.cla()
plt.plot([i for i in range(Generations)],fitnesses) #show fintesses over generations
plt.title("Results of population fitness over "+str(Generations)+" generations")
plt.ylabel("Fitness Units")
plt.xlabel("Generation")
plt.show()