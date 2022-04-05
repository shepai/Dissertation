#from agent import Agent_defineLayers as robot
from inputs import get_gamepad
import board
from adafruit_ina219 import ADCResolution, BusVoltageRange, INA219
import numpy as np
import copy
from wheg import *

i2c_bus = board.I2C()

ina1 = INA219(i2c_bus,addr=0x40)
ina2 = INA219(i2c_bus,addr=0x41)
ina3 = INA219(i2c_bus,addr=0x42)

print("ina219 test")

ina1.bus_adc_resolution = ADCResolution.ADCRES_12BIT_32S
ina1.shunt_adc_resolution = ADCResolution.ADCRES_12BIT_32S
ina1.bus_voltage_range = BusVoltageRange.RANGE_16V

ina2.bus_adc_resolution = ADCResolution.ADCRES_12BIT_32S
ina2.shunt_adc_resolution = ADCResolution.ADCRES_12BIT_32S
ina2.bus_voltage_range = BusVoltageRange.RANGE_16V

ina3.bus_adc_resolution = ADCResolution.ADCRES_12BIT_32S
ina3.shunt_adc_resolution = ADCResolution.ADCRES_12BIT_32S
ina3.bus_voltage_range = BusVoltageRange.RANGE_16V

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
    def run(self,gene,trials): #run with given gene
        start=0
        arr=[]
        for i in range(trials): #run for set amount of trial times
            data=np.array([0,0])
            self.agent.set_genes(gene)
            act=self.agent.get_action(data)
            arr.append(act)

        end=0
        return self.fitness_func(start,end,np.array(arr))
    def mutation(self,gene, mean=0, std=0.5,size=100): #mutate a specific part 
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


#agent = robot(2,[3,3],3) #define output layer 
#alg = genetic(agent,10) #get GA properties
chassis=whegbot() #get chassis control

a=[0 for i in range(10)] #define the current copying
stuck=False
movingForward=False
movingBackward=False
moving=False
while 1:
    bus_voltage1 = ina1.bus_voltage        # voltage on V- (load side)
    shunt_voltage1 = ina1.shunt_voltage    # voltage between V+ and V- across the shunt
    power1 = ina1.power
    current1 = ina1.current                # current in mA

    bus_voltage2 = ina2.bus_voltage        # voltage on V- (load side)
    shunt_voltage2 = ina2.shunt_voltage    # voltage between V+ and V- across the shunt
    power2 = ina2.power
    current2 = ina2.current                # current in mA
    
    bus_voltage3 = ina3.bus_voltage        # voltage on V- (load side)
    shunt_voltage3 = ina3.shunt_voltage    # voltage between V+ and V- across the shunt
    power3 = ina3.power
    current3 = ina3.current                # current in mA
    a.append(current2/1000) #add current current
    a.pop(0) #remove previous
    b=np.array(a.copy())
    c=np.argmax(b>=0.4)
    #print(a,c)
    if c>=3: #has been stuck for multiple runs
        stuck=True
        print("stuck")
    else:
        stuck=False
    try:
        events = get_gamepad()
    except:
        chassis.stop()
        raise OSError("...")

    for event in events:
         #left stick
         #print(event.code)
         if event.code=="BTN_TRIGGER_HAPPY1":
             if moving:
                 moving=False
                 chassis.stop()
             else:
                moving=True
                chassis.rightTurn()
         elif event.code=="BTN_TRIGGER_HAPPY2":
             if moving:
                 moving=False
                 chassis.stop()
             else:
                moving=True
                chassis.leftTurn() 
         elif event.code=="ABS_RZ":
             if event.state==0:
                 chassis.stop() 
                 movingForward=False
             else:
                 chassis.forward()
                 movingForward=True
         elif event.code=="ABS_Z":
             if event.state==0:
                 chassis.stop()
                 movingBackward=False
             else:
                 chassis.backward()
                 movingBackward=True
         elif event.code=="BTN_TRIGGER_HAPPY3":
             chassis.rotateDown()
         elif event.code=="BTN_TRIGGER_HAPPY4":
             chassis.rotateUp()
         #right stick
         elif event.code == "ABS_RX":
             print("right stick forward/backward", event.state)
         elif event.code == "ABS_RY":
             print("right stick turn", event.state)
         elif event.code!="ABS_X" and event.code!="ABS_RZ" and event.code!="ABS_Z":
             pass
    if movingForward:
        chassis.forward()
    elif movingBackward:
        chassis.backward()
        
chassis.stop()
