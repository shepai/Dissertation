import matplotlib.pyplot as plt
import copy
import math
import time
import random
import numpy as np

plt.ylim((-50,50))
plt.xlim((-50,50))

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

class end:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        
class point:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.outer=[(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        self.colours=["r","r","g","g"]
        self.connected=None
        self.show()
    def show(self):
        plt.scatter(self.x,self.y)
        c=0
        for x,y in self.outer:
            plt.plot([self.x,x], [self.y,y],c=self.colours[c])
            plt.scatter(x,y,c=self.colours[c])
            c+=1
        if self.connected!=None:
            plt.plot([self.x,self.connected.x], [self.y,self.connected.y],c="b")
    def rotate(self,angle): #rotate around angle
        a=[]
        for point in self.outer:
            a.append(rotate((self.x,self.y),point,angle))
        self.outer=copy.deepcopy(a)

        conn=self.connected
        while conn!=None:
            x=conn.x
            y=conn.y
            conn.x,conn.y=rotate((self.x,self.y),(x,y),angle)
            plt.plot([self.x,x], [self.y,y],c="b")
            po=conn.outer
            a=[]
            for point in po:
                a.append(rotate((self.x,self.y),point,angle))
            conn.outer=copy.deepcopy(a)
            conn.show()
            conn=conn.connected #next movement
        self.show()
    def connect(self,point):
        plt.plot([self.x,point.x],[self.y,point.y],c="b")
        self.connected=point


class servo:
    #virtual servo code to simulate he conditions of a servo
    def __init__(self,point,Min=-100,Max=280,ang=90):
        self.point=point
        self.min=Min
        self.max=Max
        self.angle=ang
        self.start=ang
    def rotate(self,angle):
        if angle>=self.min and angle<=self.max: #check in bounds          
            self.point.rotate(math.radians(self.angle-angle))
            self.angle=angle
        else:
            #raise ValueError("Must be within bounds")
            pass
    def show(self):
        self.point.show()

class leg:
    def __init__(self,joints=3,x_pos=10,start_y=2,distance=10): #create structure
        self.joints=[]
        self.servos=[]
        p=point(x_pos,start_y)
        start_y+=distance
        self.joints.append(p)
        for i in range(1,joints):
            p=point(x_pos,start_y)
            p.connect(self.joints[i-1])
            self.servos.append(servo(self.joints[i-1]))
            self.joints.append(p)
            start_y+=distance
        self.servos.append(servo(self.joints[joints-1]))
        
    def move(self,jointNum,ang): #move a specific servo
        self.servos[jointNum].rotate(ang)

    def update(self,epoch=0): #update the display
        plt.cla()
        plt.ylim((-50,50))
        plt.xlim((-50,50))
        plt.title(epoch)
        for i in range(len(self.servos)):
            self.servos[i].show() #show each

l=leg()
def get_distance(p1,p2): #measure distance between points
    y1,y2=p1[1],p2[1]
    x1,x2=p1[0],p2[0]
    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

def set_items(angles):
    for i in range(len(l.servos)):
        l.move(i,angles[i])

#fig, ax = plt.subplots()

l.update()
l.servos[2].point.rotate(-0.1)
l.servos[1].point.rotate(0.4)
data={}
for i in range(800): #perform
    print(i)
    randPoint=(random.randint(0,12),random.randint(2,8))
    #calculate angle between
    l.update(epoch=i)
    plt.scatter(randPoint[0],randPoint[1],c="y",s=20)
    plt.pause(0.05)
    
    d={}
    d["x"]=randPoint[0]
    d["y"]=randPoint[1]
    #get lines and coords
    s0x=l.servos[0].point.x #Point two
    s0y=l.servos[0].point.y
    s1x=l.servos[1].point.x #Point two
    s1y=l.servos[1].point.y
    s2x=l.servos[2].point.x #Point one
    s2y=l.servos[2].point.y
    if s2x==randPoint[0]: randPoint=[s2x+0.1,randPoint[1]]
    d3=get_distance([s2x,s2y],randPoint) #distance from p1 to p3
    d1=10#get_distance([s2x,s2y],[s1x,s1y]) #distance from p1 to p3
    d2=10#get_distance([s2x,s2y],[s1x,s1y]) #distance from p1 to p3
    b1=get_distance([s2x,s2y],[s0x,s0y]) #distance from p1 to p3

    #current triangle
    midpoint1=((s2x+s0x)/2,(s2y+s0y)/2)
    h=get_distance([s2x,s2y],midpoint1)
    area=(b1*h)/2

    #gather new h
    h=math.sqrt(abs(d1**2 - (d3/2)**2))
    #h=(d1*h)/d3
    #gather line equation
    m=(s2y-randPoint[1])/(s2x-randPoint[0])
    nm=-1/m
    midpoint=((s2x+randPoint[0])/2,(s2y+randPoint[1])/2)
    b=midpoint[1]-(nm*midpoint[0])
    n=5
    if n==midpoint[0]: n=4 #prevent error from same point
    y=(nm*n)+b #calculate new pair of coords
    
    #vector (5,y) and midpoint
    v=(n-midpoint[0],y-midpoint[1]) #get vector
    ux=v[0]/(math.sqrt(v[0]**2 + v[1]**2)) #xvec
    uy=v[1]/(math.sqrt(v[0]**2 + v[1]**2)) #yvec
    p2x=midpoint[0]-(h*ux) #x point
    p2y=midpoint[1]-(h*uy) #y point

    #cosine rotate joint
    d4=get_distance([s1x,s1y],[p2x,p2y]) #distance from p1 to p3
    Theta=math.acos((d4**2 - (d1**2)  - (d2**2))/(-2*d2*d1))
    
    if p2x+p2y>s1x+s1y: #move to correct side
        l.servos[2].rotate(l.servos[2].angle-math.degrees(Theta))
    else:
        l.servos[2].rotate(l.servos[2].angle+math.degrees(Theta))
    l.update(epoch=i)
    #cosine rotate knee
    s0x=l.servos[0].point.x #Point two
    s0y=l.servos[0].point.y
    s1x=l.servos[1].point.x #Point two
    s1y=l.servos[1].point.y
    d5=get_distance([s0x,s0y],randPoint) #distance from p1 to p3
    d6=get_distance([s1x,s1y],randPoint) #distance from p1 to p3
    Theta=math.acos((d5**2 - (d2**2) - (d6**2))/(-2*d6*d2))

    if s0x>randPoint[0]: #move to correct side
        l.servos[1].rotate(l.servos[1].angle+(math.degrees(Theta)))
    else:
        l.servos[1].rotate(l.servos[1].angle-(math.degrees(Theta)))
    d["base"]=l.servos[2].angle
    d["knee"]=l.servos[1].angle
    d["error"]=get_distance([s0x,s0y],randPoint)
    data[str(i)]=d #store data   
    l.update(epoch=i)
    plt.plot([s2x,p2x],[s2y,p2y])
    plt.plot([randPoint[0],p2x],[randPoint[1],p2y])
    plt.scatter(randPoint[0],randPoint[1],c="y",s=20)
    plt.scatter(p2x,p2y,c="b",s=20)

    plt.pause(0.05)

    #break
    
plt.show()

import json
with open("./d.json", 'w') as f: #make empty file
                json.dump(data, f)


"""
#Theta = math.atan2(s1y-s2y, s1x-s2x)
    
    internal=((d4**2) - (d1**2) - (d3**2))/(-2*d3*d1)
    #if internal or internal < 0: #approximate if not reachable
    #    print(internal)
    #    #internal=round(internal)
    Theta = math.acos(internal)
    print("angle1",math.degrees(Theta))
    print("d3",d3)
    print("d1",d1)
    #if internal > 1: internal=1.0
    print(internal)
    assert( internal <= 1 or internal >= -1) #approximate if not reachable
    Theta = Theta - math.acos(internal)
    l.servos[2].point.rotate(Theta)
    #print(math.degrees(Theta))
    """
