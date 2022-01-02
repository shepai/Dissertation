import matplotlib.pyplot as plt
import math
import time
import random
import copy

import numpy as np
import copy
import torch

class Agent:
    def __init__(self, num_input, num_hiddenLayer, num_hiddenLayer2,num_hiddenLayer3, num_output):
        self.num_input = num_input  #set input number
        self.num_output = num_output #set ooutput number
        self.num_hidden=num_hiddenLayer
        self.num_hidden2=num_hiddenLayer2
        self.num_hidden3=num_hiddenLayer3
        self.num_genes = (num_input * num_hiddenLayer) + (num_hiddenLayer * num_hiddenLayer2) + (num_hiddenLayer2 * num_hiddenLayer3) + (num_hiddenLayer3*num_output)+num_output
        self.weights = None
        self.weights2=None
        self.weights3=None
        self.weights4=None
        self.bias = None
        print("Manual",self.num_genes)
    def set_genes(self, gene):
        weight_idxs = self.num_input * self.num_hidden #size of weights to hidden
        weights2_idxs = self.num_hidden * self.num_hidden2 + weight_idxs #size and position
        weights3_idxs = self.num_hidden2 *self.num_hidden3 + weights2_idxs #weight_idxs #size and position
        weights4_idxs = self.num_hidden3 * self.num_output + weights3_idxs
        bias_idxs = weights4_idxs+ self.num_output #sizes of biases
        w = gene[0 : weight_idxs].reshape(self.num_hidden, self.num_input)   #merge genes
        w2 = gene[weight_idxs : weights2_idxs].reshape(self.num_hidden2, self.num_hidden)   #merge genes
        w3 = gene[weights2_idxs: weights3_idxs].reshape(self.num_hidden3, self.num_hidden2) 
        w4 = gene[weights3_idxs: weights4_idxs].reshape(self.num_output, self.num_hidden3) 
        b = gene[weights4_idxs: bias_idxs].reshape(self.num_output,) #merge genes
        self.weights = torch.from_numpy(w) #assign weights
        self.weights2 = torch.from_numpy(w2) #assign weights
        self.weights3 = torch.from_numpy(w3) #assign weights
        self.weights4 = torch.from_numpy(w4) #assign weights
        self.bias = torch.from_numpy(b) #assign biases

    def forward(self, x):
        x = torch.from_numpy(x.flatten()/255).unsqueeze(0)
        x = torch.mm(x, self.weights.T) #first layer
        x =torch.mm(x,self.weights2.T) #second layer
        x =torch.mm(x,self.weights3.T)
        return torch.mm(x,self.weights4.T) + self.bias #third layer
        
    def get_action(self, x):
        vectors=[(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)] #possible moves
        arr=list(self.forward(x)[0])
        ind=np.argmax(arr)
        return vectors[ind]


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
        weights_idxs=[]
        for i in range(len(self.hidden)-1):
            current+=self.hidden[i]*self.hidden[i+1]
            weights_idxs.append(current)
        #print(weights_idxs[-1])
        weights_idxs.append(self.hidden[-1] * self.num_output + weights_idxs[-1])
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