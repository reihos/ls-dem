# Updating the location of a bouncing ball using molecular dynamic concepts and comparing it with the results from LAMMPS

import numpy as np
import math 
import matplotlib.pyplot as plt
import json

# SI Units

# Particle properties
m=0.0654498 #based on the diameter and a density of 1000 kg/m^3
diam=0.05
r=diam/2

# Contact properties and initial conditions
Kn=10000.
Cn=5.772 # calculated by assuming a COR of 0.7 (O'Sullivan 2011)

H0=0.2
g=9.81

# Geometrical properties
ywall=0.
ytop=0.25
ybottom=-0.05

# Discretization
dy=0.01
dt=0.000001
n=4 #number of boundary nodes for the particle
alpha=2*math.pi/n

# Initializing t,y and v arrays
t=np.arange(0,1,dt)
y=np.zeros((np.size(t),n+1))
y[0,0]=H0
for i in range(1,n+1):
    y[0,i]=y[0,0]+r*math.sin(alpha*(i-1))
v=np.zeros(np.size(t))

# Defining the level set function for the wall
ny=int((ytop-ybottom)/dy)
LS=np.zeros(ny)
for i in range(0,ny):
    LS[i]=ybottom+i*dy
ibottom=ybottom/dy #needed in the next step

# Updating Particle Location
for i in range(0,np.size(t)-1):
    a=-g
    # finding the level set function at every node on the particle
    # and updating the acceleration if LS<0
    for j in range(1,n+1):
        yfloor=math.floor(y[i,j]*100)/100
        ifloor=int(math.floor(y[i,j]/dy)-ibottom)
        LSi=LS[ifloor]+(y[i,j]-yfloor)/dy*(LS[ifloor+1]-LS[ifloor])
        if LSi<0: 
            a=a+(-Kn*LSi-Cn*v[i])/m

    v[i+1]=v[i]+a*dt
    # updating the location of the center of the particle
    y[i+1,0]=y[i,0]+v[i]*dt+0.5*a*dt*dt
    # updating the location of all the nodes on the particle
    for j in range(1,n+1):
        y[i+1,j]=y[i+1,0]+r*math.sin(alpha*(j-1))


# Theoretical Solution

# oscillator parameters    
Ccrit=2*math.sqrt(Kn*m)
zeta=Cn/Ccrit
wn=math.sqrt(Kn/m)
wd=wn*math.sqrt(1-zeta**2)

# initiation
yth=np.zeros(np.size(t))
yth[0]=H0
v0=0
t0=0
h0=H0
flag=0

# calculation
for i in range(0,np.size(t)-1):
    if yth[i]>=r:
        if flag==1:
            v0=math.exp(-zeta*wn*(t[i]-tcol))*(v0col*math.cos(wd*(t[i]-tcol))-zeta/math.sqrt(1-zeta**2)*v0col*math.sin(wd*(t[i]-tcol)))
            t0=t[i]
            h0=yth[i]
            flag=0

        yth[i+1]=-0.5*g*(t[i+1]-t0)**2+v0*(t[i+1]-t0)+h0

    else:
        if flag==0:
            v0col=v0-g*(t[i]-t0)
            tcol=t[i]
            ycol=yth[i]
            flag=1

        u=math.exp(-zeta*wn*(t[i+1]-tcol))*(v0col/wd*math.sin(wd*(t[i+1]-tcol)))-m*g/Kn
        yth[i+1]=ycol+u


# Importing the LAMMPS results
with open("lammps.json","r") as f:
    data=json.load(f)

# Plotting
plt.figure(figsize=(10,7))
plt.plot(t,y[:,0],label="LS-DEM")
plt.plot(np.arange(len(data["height"]))*data["dt"],data["height"],"--r",label="LAMMPS")
plt.plot(t,yth,":k",label="Theoretical")
plt.legend()
plt.show()

