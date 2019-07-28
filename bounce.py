# written by Rei Hosseini 07/08/2019
# Updating the location of a bouncing ball using molecular dynamic concepts and comparing it with the results from LAMMPS

import numpy as np
import math 
import matplotlib.pyplot as plt
import json


# particle properties
m=0.0654498 #based on the diameter and a density of 1000 kg/m^3
diam=0.05
r=diam/2

# contact properties and initial conditions
Kn=10000.
Cn=5.772 # calculated by assuming a COR of 0.7 (O'Sullivan 2011)
xwall=0.

H0=0.2
g=9.81

dt=0.000001

# Initializing t,x and v arrays
t=np.arange(0,1,dt)
x=np.zeros(np.size(t))
x[0]=H0
v=np.zeros(np.size(t))

# Updating Particle Location
for i in range(0,np.size(t)-1): 
    dn=x[i]-xwall-r
    if dn>=0:
        a=-g
    else:
        a=max((-Kn*(x[i]-r)-Cn*v[i]),0)/m-g
    
    v[i+1]=v[i]+a*dt
    #x[i+1]=(v[i]+v[i+1])/2*dt+x[i]
    x[i+1]=x[i]+v[i]*dt+0.5*a*dt*dt
np.savetxt("bounce.csv",x,delimiter=",")


# Loading JSON results of LAMMPS
with open("lammps.json","r") as f:
    data=json.load(f)

# Plotting
plt.plot(t,x,label="My Code")
plt.plot(np.arange(len(data["height"]))*data["dt"],data["height"],"--r",label="LAMMPS")
plt.legend()
plt.show()

