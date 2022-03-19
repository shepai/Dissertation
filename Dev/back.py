from agent import Agent_defineLayers as robot
import numpy as np
import copy

agent = robot(2,[3,3],2) #define output layer 

class genetic:
    def __init__(self,agent,pop_size):
        self.agent=agent
        gene_pop=[]
        for i in range(pop_size): #vary from 10 to 20 depending on purpose of robot
            gene=np.random.normal(0, 0.5, (agent.num_genes))
            gene_pop.append(copy.deepcopy(gene))#create
    def fitness_func(self):
        pass
    def run_microbial(self,gen=50):
        assert type(gen)==type(50),"Generations must be an integer"
        for epoch in len(gen):
            pass

alg = genetic(agent,10)