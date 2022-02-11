import matplotlib.pyplot as plt
import math
import time
import random
import copy

import numpy as np
import copy
import torch

class Agent_defineLayers:
    def __init__(self, num_input, layers, num_output):
        assert type(layers)==type([]), "Error with layers, give array of the number of layers"
        self.num_input = num_input  #set input number
        self.num_output = num_output #set ooutput number
        self.hidden=[]
        last=num_input
        self.num_genes=0
        for layer in layers:
            self.hidden.append(layer)
            self.num_genes+=(last * layer)
            last=layer
        self.num_genes +=(self.hidden[-1]*num_output)+num_output
        self.weights = None
        self.hidden_weights=None
        self.bias = None
        print("Auto",self.num_genes)
    def set_genes(self, gene):
        weight_idxs = self.num_input * self.hidden[0] #size of weights to hidden
        current=weight_idxs
        weights_idxs=[current] #start with end of last
        for i in range(len(self.hidden)-1):
            current+=self.hidden[i]*self.hidden[i+1] #calculate next idx for each layer
            weights_idxs.append(current)
        bias_idxs=None
        weights_idxs.append(self.hidden[-1] * self.num_output + weights_idxs[-1]) #add last layer heading to output
        bias_idxs = weights_idxs[-1]+ self.num_output #sizes of biases
        w = gene[0 : weight_idxs].reshape(self.hidden[0], self.num_input)   #merge genes
        ws=[]
        for i in range(len(self.hidden)-2):
            ws.append(gene[weights_idxs[i] : weights_idxs[i+1]].reshape(self.hidden[i+1], self.hidden[i]))
        ws.append(gene[weights_idxs[-2] : weights_idxs[-1]].reshape(self.num_output, self.hidden[-1]))
        b = gene[weights_idxs[-1]: bias_idxs].reshape(self.num_output,) #merge genes

        self.weights = torch.from_numpy(w) #assign weights
        self.hidden_weights=[]
        for w in ws:
            self.hidden_weights.append(torch.from_numpy(w))
        self.bias = torch.from_numpy(b) #assign biases

    def forward(self, x):
        x = torch.from_numpy(x.flatten()/255).unsqueeze(0)
        x = torch.mm(x, self.weights.T) #first layer
        for i in range(len(self.hidden_weights)-1):
            x =torch.mm(x,self.hidden_weights[i].T) #second layer
        return torch.mm(x,self.hidden_weights[-1].T) + self.bias #third layer
        
    def get_action(self, x):
        vectors=[(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)] #possible moves
        arr=list(self.forward(x)[0])
        ind=np.argmax(arr)
        return vectors[ind]

class Agent_Conv:
    def __init__(self, num_input, layers, num_output):
        assert type(layers)==type([]), "Error with layers, give array of the number of layers"
        self.num_input = num_input  #set input number
        self.num_output = num_output #set ooutput number
        self.hidden=[]
        last=num_input
        self.num_genes=0
        for layer in layers:
            self.hidden.append(layer)
            self.num_genes+=(last * layer)
            last=layer
        self.num_genes +=(self.hidden[-1]*num_output)+num_output
        self.weights = None
        self.hidden_weights=None
        self.bias = None
        print("Auto",self.num_genes)
    def set_genes(self, gene):
        weight_idxs = self.num_input * self.hidden[0] #size of weights to hidden
        current=weight_idxs
        weights_idxs=[current] #start with end of last
        for i in range(len(self.hidden)-1):
            current+=self.hidden[i]*self.hidden[i+1] #calculate next idx for each layer
            weights_idxs.append(current)
        bias_idxs=None
        weights_idxs.append(self.hidden[-1] * self.num_output + weights_idxs[-1]) #add last layer heading to output
        bias_idxs = weights_idxs[-1]+ self.num_output #sizes of biases
        w = gene[0 : weight_idxs].reshape(self.hidden[0], self.num_input)   #merge genes
        ws=[]
        for i in range(len(self.hidden)-2):
            ws.append(gene[weights_idxs[i] : weights_idxs[i+1]].reshape(self.hidden[i+1], self.hidden[i]))
        ws.append(gene[weights_idxs[-2] : weights_idxs[-1]].reshape(self.num_output, self.hidden[-1]))
        b = gene[weights_idxs[-1]: bias_idxs].reshape(self.num_output,) #merge genes

        self.weights = torch.from_numpy(w) #assign weights
        self.hidden_weights=[]
        for w in ws:
            self.hidden_weights.append(torch.from_numpy(w))
        self.bias = torch.from_numpy(b) #assign biases

    def forward(self, x):
        m = torch.nn.Conv1d(len(x), len(self.hidden_weights[0]), 1, stride=2)
        x = np.reshape(torch.from_numpy(x.flatten()/255).unsqueeze(0),(1,27,1))
        x = m(torch.tensor(x))
        print(type(x))
        #x = torch.mm(x, self.weights.T) #first layer
        for i in range(len(self.hidden_weights)-1):
            x =torch.mm(x,self.hidden_weights[i].T) #second layer
        return torch.mm(x,self.hidden_weights[-1].T) + self.bias #third layer
        
    def get_action(self, x):
        vectors=[(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)] #possible moves
        arr=list(self.forward(x)[0])
        ind=np.argmax(arr)
        return vectors[ind]