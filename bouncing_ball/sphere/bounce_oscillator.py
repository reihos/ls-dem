# written by Rei Hosseini 07/08/2019
# Updating the particle location using SDF oscillator equation
# Only works for the first bounce
# It was not extended to account for the next bounces dues to complexity and the fact that a simpler approach exists 

import numpy as np
import math 
import matplotlib.pyplot as plt

# particle properties
m=0.1
diam=0.05
r=diam/2

# contact properties and initial conditions
Kn=10000
Cn=6

H0=0.2
g=9.81

u0=m*g/Kn
v0=-math.sqrt(2*g*(H0-diam))

Ccrit=2*math.sqrt(Kn*m)
zeta=Cn/Ccrit
wn=math.sqrt(Kn/m)
wd=wn*math.sqrt(1-zeta**2)
T=2*math.pi/wd

t1=math.sqrt(2*(H0-diam/2)/g)

dt=0.0001
t=np.arange(0,0.5,dt)
x=np.zeros(np.size(t))
x[0]=H0
v=np.zeros(np.size(t))

# Updating the location of the particle
for i in range(0,np.size(t)-1):
    v[i+1]=v[i]-g*dt
    if x[i]>r: 
        x[i+1]=(v[i]+v[i+1])/2*dt+x[i]
    else:
        dtcol=t[i+1]-t1
        u=math.exp(-zeta*wn*dtcol)*(u0*math.cos(wd*dtcol)+(v0+zeta*wn*u0)/wd*math.sin(wd*dtcol))-m*g/Kn
        x[i+1]=r+u

plt.plot(t,x)
plt.show()
