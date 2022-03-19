from agent import Agent_defineLayers as robot
import numpy as np
import copy


class genetic:
    def __init__(self,agent,pop_size):
        self.agent=agent
        self.gene_pop=[]
        for i in range(pop_size): #vary from 10 to 20 depending on purpose of robot
            gene=np.random.normal(0, 0.5, (agent.num_genes))
            self.gene_pop.append(copy.deepcopy(gene))#create
    def fitness_func(self,startDist,endDist,movement):
        std=(np.std(movement) + 0.1) * 0.1 # gather standard deviation. The lower the better. 10% weighting. not allowed to be 0
        moved = startDist - endDist
        if moved<=20:
            return 0
        return max(100-(std*moved),0)
    def select_genes(self,k=1):
        #use k to evolve neighbours
        index=np.random.randint(0,len(self.gene_pop)-1)
        neighbour_idx = np.random.randint(max(0,index-k),min(len(self.gene_pop)-1,index+k))
        return copy.deepcopy(self.gene_pop[index]),copy.deepcopy(self.gene_pop[neighbour_idx]),index,neighbour_idx
    def place_genes(self,index1,index2,gene1,gene2):
        self.gene_pop[index1]=copy.deepcopy(gene1)
        self.gene_pop[index2]=copy.deepcopy(gene2)
    def run(self,gene,trials):
        start=0
        arr=[]
        for i in range(trials): #run for set amount of trial times
            data=np.array([0,0])
            self.agent.set_genes(gene)
            act=self.agent.get_action(data)
            arr.append(act)

        end=0
        return self.fitness_func(start,end,np.array(arr))
    def mutation(self,gene, mean=0, std=0.5,size=100):
        assert size<len(gene)
        n=np.random.randint(0,len(gene)-size-1)
        array=np.random.normal(mean,std,size=size)
        gene = gene[n:n+size] + array #mutate the gene via normal 
        # constraint
        gene[gene >4] = 4
        gene[gene < -4] = -4
        return gene
    def run_microbial(self,gen=50):
        assert type(gen)==type(50),"Generations must be an integer"
        for epoch in len(gen):
            gene1,gene2,id1,id2=self.select_genes()
            #gather sensor data
            #gather prediction

            input("Press ENTER to start trial")
            fitness1=self.run(gene1)

            input("Press ENTER to start trial")
            fitness2=self.run(gene2)

            #assign selection
            W,L=None,None
            if fitness1>fitness2: 
                W=gene1
                L=gene2
            else:
                L=gene1
                W=gene2

            L=copy.deepcopy(self.mutate(W,size=9))  #mutate winner and place back

            self.place_genes(id1,id2,W,L) #palce back into the pop


agent = robot(2,[3,3],3) #define output layer 
alg = genetic(agent,10)