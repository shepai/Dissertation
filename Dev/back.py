from agent import Agent_defineLayers as robot
import numpy as np
import copy

agent = robot(2,[3,3],2) #define output layer 

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
        neighbour_ids = np.random.randint(max(0,index-k),min(len(self.gene_pop)-1,index+k))

    def run_microbial(self,gen=50):
        assert type(gen)==type(50),"Generations must be an integer"
        for epoch in len(gen):
            input("Press ENTER to start trial")
            #gather sensor data
            #gather prediction

alg = genetic(agent,10)