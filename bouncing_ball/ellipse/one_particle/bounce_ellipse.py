# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math 
import matplotlib.pyplot as plt
import json

# SI Units

# Particle properties
m=0.0654498
a=0.05 #ellipse
b=0.025
I=m/4*(a**2+b**2)

# Contact properties and initial conditions
Kn=10000.
Cn=5.772 
#Cn=0.
H0=0.2
g=9.81
theta=20/180*math.pi #angle of the ball

# Geometrical properties
ywall=0.
ytop=0.65
ybottom=-0.05

# Discretization
dy=0.01
dt=0.000001
n=32 #number of boundary nodes for the particle
d_angle=2*math.pi/n #discretization angle

# Initializing t, x, y, v, omega, and energy arrays
t=np.arange(0,1,dt)

xc=np.zeros(len(t)) # x of center
yc=np.zeros(len(t)) # y of center
yc[0]=H0
x=np.zeros((len(t),n)) 
y=np.zeros((len(t),n))

r=np.zeros(n)
for j in range(0,n):
    r[j]= a*b/math.sqrt((b*math.cos(d_angle*j))**2+(a*math.sin(d_angle*j))**2)
for j in range(0,n):
    x[0,j]=xc[0]+r[j]*math.cos(d_angle*(j)+theta)
    y[0,j]=yc[0]+r[j]*math.sin(d_angle*(j)+theta)

v=np.zeros(len(t)) #velocity of the center of mass
vy=np.zeros((len(t),n)) #velocity of each point on the particle
omega=0.

a=np.zeros(len(t))
alpha=np.zeros(len(t))

Ep=np.zeros(len(t)-1)
Ek=np.zeros(len(t)-1)
Et=np.zeros(len(t)-1)

# Defining the level set function for the wall
ny=int((ytop-ybottom)/dy)
LS=np.zeros(ny)
for i in range(0,ny):
    LS[i]=ybottom+i*dy
ibottom=ybottom/dy #needed in the next step
LSi=np.zeros(n)

# Updating Particle Location
for i in range(0,len(t)-1):
    a[i]=-g
    Es=0
    # finding the level set function at every node on the particle
    # and updating the acceleration if LS<0
    for j in range(0,n):
        yfloor=math.floor(y[i,j]*100)/100
        ifloor=int(math.floor(y[i,j]/dy)-ibottom)
        LSi[j]=LS[ifloor]+(y[i,j]-yfloor)/dy*(LS[ifloor+1]-LS[ifloor])
        if LSi[j]<0: 
            a[i]=a[i]+(-Kn*LSi[j]-Cn*vy[i,j])/m #linear acceleration in the y direction
            alpha[i]=alpha[i]+(-Kn*LSi[j]-Cn*vy[i,j])*(x[i,j]-xc[i])/I #angular acceleration
            Es+=0.5*Kn*LSi[j]**2

    # updating the location of the center of the particle
    yc[i+1]=yc[i]+v[i]*dt+0.5*a[i]*dt**2
    xc[i+1]=xc[i]
    # calculating the rotation angle
    theta+=omega*dt+0.5*alpha[i]*dt**2
    # updating the location of all the nodes on the particle
    for j in range(0,n):
        x[i+1,j]=xc[i+1]+r[j]*math.cos(d_angle*j+theta)
        y[i+1,j]=yc[i+1]+r[j]*math.sin(d_angle*j+theta)
    
    # updating the velocities
    v[i+1]=v[i]+a[i]*dt
    omega+=alpha[i]*dt
    for j in range(0,n):
        vy[i+1,j]=vy[i,j]+a[i]*dt+alpha[i]*dt*r[j]*math.cos(d_angle*j+theta)
    # caculating the energies
    Ep[i]=m*g*yc[i]+Es
    Ek[i]=0.5*m*v[i]**2+0.5*I*omega**2
    Et[i]=Ep[i]+Ek[i]

    # printing
    printing=0
    if printing==1 and i%1==0:
        print("step number",i)
        for j in range(0,n):
            print("node = ", j, "LS = ", LSi[j], x[i,j])
        print( "a = ", a[i], "alpha = ", alpha[i])
        print("x,y of center", xc[i+1],yc[i+1], "theta" , theta)
        print("v = ", v[i+1],"vy[10]", vy[i+1,10],"omega = ", omega)



# Importing the LAMMPS results
with open("../../sphere/lammps.json","r") as f:
    data=json.load(f)

# Plotting
plot='12'
if '1' in plot:
    plt.figure(figsize=(10,7))
    plt.plot(t,yc,label="LS-DEM")
    plt.plot(np.arange(len(data["height"]))*data["dt"],data["height"],"--r",label="LAMMPS")
    plt.legend()
    plt.xlabel("time (sec)")
    plt.ylabel("height (m)")
    plt.show()

if '2' in plot:
    plt.figure(figsize=(10,7))
    plt.plot(t[:-1],Ep,label="potential energy")
    plt.plot(t[:-1],Ek,label="kinetic energy",linestyle="--")
    plt.plot(t[:-1],Et,label="total energy")
    plt.xlabel("time (sec)")
    plt.ylabel("energy (J)")
    plt.legend()
    plt.show()

if '3' in plot:
    plt.figure()
    for n in range(100000,1000000,5000):
        plt.cla()
        plt.plot(np.append(x[n,:],x[n,0]),np.append(y[n,:],y[n,0]),'k')
        plt.axis([-0.1, 0.1, 0, 0.2])
        plt.gca().set_aspect('equal')
        plt.savefig('Frame%07d.png' %n)

if '4' in plot:
    fig, axs=plt.subplots(3)
    axs[0].plot(t,a,'k')
    axs[0].set(ylabel=r"$a\ (m/s^2)$")
    axs[1].plot(t,alpha,'k')
    axs[1].set(ylabel=r"$\alpha\ (rad/s^2)$")
    axs[2].plot(t,v,'k')
    axs[2].set(ylabel=r"$v\ (m/s)$")
    plt.xlabel("time (sec)")
    xlim=[0.18,0.22]
    for ax in axs:
        ax.set_xlim(xlim)
    plt.show()
    
    
