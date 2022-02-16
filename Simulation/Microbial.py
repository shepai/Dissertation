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

def mutation(gene, mean=0, std=0.5,size=100):
    assert size<len(gene)
    n=random.randint(0,len(gene)-size-1)
    array=np.random.normal(mean,std,size=size)
    gene = gene[n:n+size] + array #mutate the gene via normal 
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
        x1=((-1*bx)+math.sqrt((bx**2)-(4*c)))/(2)
        x2=((-1*bx)-math.sqrt((bx**2)-(4*1*c)))/(2)
        coords.append([x1,y])
        coords.append([x2,y])
    return coords
def run_trial(gene,runs=30):
    pathx=[]
    pathy=[]
    startPos=pickPosition(world,4,LBounds=6)
    current=startPos.copy()
    energy=0
    last=startPos.copy()
    broke=False
    routeValues=[]
    v=rnd.choice(vectors)
    whegBot.set_genes(gene) #set the genes of the agent
    radius=10
    valid=False
    cords=[]
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
    while i<runs and not broke and getDist(current,cords)>1: #loop through and generate path
        dir=maths.cos(v[1]) #get angle from y-axis
        im=readIm(world,current,dir) #read the image that the agent sees
        assert len(im)==25, "Panoramic Camera failed"+str(len(im)) #assert length correct
        VectorBetween=[cords[0]-current[0],cords[1]-current[1]]
        v=whegBot.get_action(np.concatenate((im, VectorBetween))) #get action from the agent
        last=current.copy() 
        pathx.append(current[0]+v[0])
        pathy.append(current[1]+v[1])
        lastH=world[current[1]][current[0]]
        current[0]+=v[0]
        current[1]+=v[1]
        if current[0]>=0 and current[0]<len(world[0])-1 and current[1]>=0 and current[1]<len(world)-1:
            if world[current[1]][current[0]]<=-6 or lastH-world[current[1]][current[0]]>3: #do not allow the rover to enter water
                #or if there is a substanial step
                broke=True
            else: #calculate energy usage
                climb=max(-1,world[current[1]][current[0]]-world[last[1]][last[0]]) #non 0 value of total climb
                routeValues.append(abs(climb))
                energy+=1+climb
        i+=1
    endDist=getDist(current,cords)
    #print("total energy consumed",energy,"fitness",fitness(broke,energy,endDist),"endDist:",endDist)
    
    return pathx,pathy,fitness(broke,energy,endDist),cords

def microbial(genes,world,position):
    global BESTFIT
    global BEST
    #microbial algorithm trial
    ind_1 = rnd.randint(0,len(genes)-1)
    ind_2=0
    if ind_1>0: ind_2 = ind_1-1
    else: ind_2= ind_1+1
    #while ind_1==ind_2: #make value unique
    #    ind_2 = rnd.randint(0,len(genes)-1)
    #get two random positions
    gene1=(genes[ind_1])
    gene2=(genes[ind_2])
    #run trial for each
    p1x,p1y,fitness1,endCord1=run_trial(gene1)
    p2x,p2y,fitness2,endCord2=run_trial(gene2)

    #run same trial again to generate different coord
    a,b,fitnessa1,c=run_trial(gene1)
    a,b,fitnessa2,c=run_trial(gene2)
    a,b,fitnessb1,c=run_trial(gene1)
    a,b,fitnessb2,c=run_trial(gene2)

    #generate average fitness
    fitness1=(fitness1+fitnessa1+fitnessb1)/3
    fitness2=(fitness2+fitnessa2+fitnessb2)/3
    #print(max(fitness1,fitness2),"% over 3 trials")
    #show results
     
    #microbial selection
    if fitness1>fitness2:
        gene2=copy.deepcopy((gene1)) #crossover
        genes[ind_2]=copy.deepcopy(crossover(gene2,mutation(gene1)))
    elif fitness2>fitness1:
        gene1=copy.deepcopy((gene2)) #crossover
        genes[ind_1]=copy.deepcopy(crossover(gene1,mutation(gene2)))
    if max(fitness1,fitness2)>BESTFIT:
        BESTFIT=max(fitness1,fitness2) #gather the maximum
        if fitness1>fitness2: BEST=[copy.deepcopy(p1x),copy.deepcopy(p1y),copy.deepcopy(world),endCord1]
        else: BEST=[copy.deepcopy(p2x),copy.deepcopy(p2y),copy.deepcopy(world),endCord2]
    if BESTFIT==0: BEST=[copy.deepcopy(p2x),copy.deepcopy(p2y),copy.deepcopy(world),endCord2] #default
    
    return genes,max(fitness1,fitness2)
BEST=[]
BESTFIT=-1
world,shape=generateWorld()
startPos=[int(SIZE/2),int(SIZE/2)] #centre point

testIm=readIm(world,[25,25],30) #read the image that the agent sees
Generations=1000
vectors=[(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)] #possible moves
#network input:
#   image, x_dest, y_dest
#layer 1
#   10
#layer 2
#   10
#output
#   vectors possible (8)
whegBot=Agent_defineLayers(testIm.shape[0]+2,[10,10],len(vectors)) #define the agent

pop_size=15
gene_pop=[]
for i in range(pop_size): #vary from 10 to 20 depending on purpose of robot
    gene=np.random.normal(0, 0.5, (whegBot.num_genes))
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
    
    #genes have been selected
    gene_pop,fit=microbial(gene_pop,world,startPos)
    fitnesses.append(max([fit]+fitnesses))
    print(max(fitnesses))
bestGene=[]
bestFit=0
for gene in gene_pop:
    startPos=pickPosition(world,4,LBounds=6)
    p1x,p1y,fit,endCord1=run_trial(gene)
    if fit>0:
        plt.plot(p1x,p1y) #show best path
        plt.title("Gene "+str(fit)+"% after generations")
        plt.scatter(endCord1[0],endCord1[1])
        plt.scatter(p1x[0],p1y[0],c="r")
        #print(canReach(Rmap,startPos,endPos))
        plt.imshow((BEST[2]),cmap='terrain') #show best show
        plt.show()
        if fit>bestFit:
            bestFit=fit
            bestGene=copy.deepcopy(gene)
if bestGene!=[]:
    print("How best performs",bestGene)
    for i in range(5):
        startPos=pickPosition(world,4,LBounds=6)
        p1x,p1y,fit,endCord1=run_trial(bestGene)
        plt.plot(p1x,p1y) #show best path
        plt.title("The best trials "+str(fit)+"% after generations")
        plt.scatter(endCord1[0],endCord1[1])
        plt.scatter(p1x[0],p1y[0],c="r")
        #print(canReach(Rmap,startPos,endPos))
        plt.imshow(BEST[2],cmap='terrain') #show best show
        plt.show()

np.save("microbialPart.npy", fitnesses)
"""
plt.plot(BEST[0],BEST[1]) #show best path
plt.title("Results of best fitness at "+str(BESTFIT)+"% after generations")
plt.scatter(BEST[3][0],BEST[3][1])
plt.scatter(BEST[0][0],BEST[1][0],c="r")
#print(canReach(Rmap,startPos,endPos))
plt.imshow(BEST[2],cmap='terrain') #show best show
plt.show()
"""
plt.cla()
plt.plot([i for i in range(Generations)],fitnesses) #show fintesses over generations
plt.title("Results of population fitness over "+str(Generations)+" generations")
plt.ylabel("Fitness Units")
plt.xlabel("Generation")
plt.show()


"""
if fitness(broke,energy,endDist)>0:
        plt.plot(pathx,pathy) #show best path
        plt.scatter(cords[0],cords[1])
        plt.scatter(startPos[0],startPos[1],c="y")
        plt.scatter(pathx[0],pathy[0],c="r")
        #print(canReach(Rmap,startPos,endPos))
        plt.imshow(world,cmap='terrain') #show best show
        plt.show()
"""