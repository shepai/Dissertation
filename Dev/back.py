#from agent import Agent_defineLayers as robot
from inputs import get_gamepad
import board
from adafruit_ina219 import ADCResolution, BusVoltageRange, INA219
import numpy as np
import copy
from wheg import *
from agent import *
import board
import adafruit_mpu6050
import time

i2c = board.I2C()  # uses board.SCL and board.SDA
mpu = adafruit_mpu6050.MPU6050(i2c)

class Gyro:
    def __init__(self,mpu=mpu):
        self.mpu=mpu
        self.a=0
        self.getZero()
    def getAcc(self):
        return self.mpu.acceleration
    def getGyro(self):
        return self.mpu.gyro
    def getTemp(self):
        return self.mpu.temperature
        
    def getZero(self): #create a zerod value to detect change
        self.a=0
        for i in range(200):
            self.a+=int(self.getGyro()[2]*10)
        self.a=int(self.a/200)
    def movementDetected(self,sensitivity=150):
        val=abs(int(self.getGyro()[2]*10)-self.a) #get thresholded value
        data=""
        if val>sensitivity: #chec significance
            data = "movement"
        self.getZero() #reevaluate movement threshold
        print(str(val)+data)
        return data


tilt=Gyro()
print("current tilt",tilt.getGyro())

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
        self.fitneses=[0 for i in range(pop_size)]
    def fitness_func(self,startDist,endDist,movement):
        if getStuck(): #boolean solved or didn't solve
            return 0
        return 1
        """
        std=(np.std(movement) + 0.1) * 0.1 # gather standard deviation. The lower the better. 10% weighting. not allowed to be 0
        moved = startDist - endDist
        if moved<=20:
            return 0
        return max(100-(std*moved),0)"""
    def select_genes(self,k=1):
        #use k to evolve neighbours
        index=np.random.randint(0,len(self.gene_pop)-1)
        neighbour_idx = np.random.randint(max(0,index-k),min(len(self.gene_pop)-1,index+k))
        return copy.deepcopy(self.gene_pop[index]),copy.deepcopy(self.gene_pop[neighbour_idx]),index,neighbour_idx
    def place_genes(self,index1,index2,gene1,gene2):
        self.gene_pop[index1]=copy.deepcopy(gene1)
        self.gene_pop[index2]=copy.deepcopy(gene2)
    def run(self,gene,trials,chassis): #run with given gene
        start=0
        arr=[]
        for i in range(trials): #run for set amount of trial times
            servoAng=chassis.getBack()
            x=tilt.getGyro()[0]
            data=np.array([servoAng,x])
            self.agent.set_genes(gene)
            act=self.agent.get_action(data)
            arr.append(act)
        for act in arr:
            if act==0: #up
                chassis.rotateUp()
            elif act==1: #stay
                pass
            elif act==2: #down
                chassis.rotateDown()
            time.sleep(1)
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
    def run_hillclimber(self,chassis): #hillclimber algorithm
        gene1,__,id1,__=self.select_genes()
        fit=self.run(gene1,3,chassis)
        if fit>self.fitneses[id1]:
            self.gene_pop[id1]=copy.deepcopy(gene1)
            self.fitneses[id1]=fit
        else: #mutate if not fixed
            self.gene_pop[id1]=copy.deepcopy(self.mutation(gene1))


a=[0 for i in range(10)] #define the current copying
def getStuck():
    global a

    current1 = ina1.current                # current in mA
    current2 = ina2.current                # current in mA
    current3 = ina3.current                # current in mA

    a.append(current2/1000) #add current current
    a.pop(0) #remove previous
    b=np.array(a.copy())
    c=np.argmax(b>=0.4)
    return True if c>3 else False

agent = Agent_defineLayers(2,[3,3],3) #define output layer 
alg = genetic(agent,10) #get GA properties
chassis=whegbot() #get chassis control

stuck=False
movingForward=False
movingBackward=False
moving=False
while 1:
    stuck=True
    while stuck: #unstick itself
        stuck=getStuck()
        #print(a,c)
        if stuck: #evolve back
            actions=alg.run_hillclimber(chassis)
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
