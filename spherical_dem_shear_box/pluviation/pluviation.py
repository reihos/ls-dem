# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math 
import matplotlib.pyplot as plt

# SI Units

# Particle properties
r=[]
f=open('radius_999_particles.txt','r')
lines=f.read().splitlines()
for line in lines:
    r.append(float(line))
np.random.shuffle(r)
r_avg=np.average(r)
total_nps=len(r)
r=np.array(r)
m=0.04

# Contact properties 
Kn=1e5
damping=0.5
Ccrit=2*np.sqrt(Kn*m)
Cn=damping*Ccrit

# Initial Conditions
H0=0.8
g=9.81

# Geometrical properties
ybottom=0.
ytop=1.
xright=1.
xleft=0.

# Discretization
dt=.0001

# Drop properties
tdrop=0.1
idrop=int(tdrop/dt)
spacing=r_avg

# Initializing t, x, y, v, omega, and energy arrays
# time
T=5
t=np.arange(0,T,dt)

# location
xc=np.zeros((len(t),total_nps)) # x of center
yc=np.zeros((len(t),total_nps)) # y of center

# velocity
vx=np.zeros((len(t),total_nps)) #velocity of the center of mass
vy=np.zeros((len(t),total_nps)) 
vx_half=np.zeros(total_nps)
vy_half=np.ones(total_nps)*g*dt/2
omega=np.zeros(total_nps)
omega_half=np.zeros(total_nps)

# acceleration
ax=np.zeros((len(t),total_nps))
ay=np.zeros((len(t),total_nps))
alpha=np.zeros((len(t),total_nps))

nps=np.zeros(len(t),dtype=int)

# Updating the Kinematic Properties of Particles
for i in range(len(t)-1):
    print(i)
    if i%idrop==0 and nps[i]<total_nps:
        H=H0+2*r_avg*t[i]/tdrop
        for j in range(nps[i], total_nps):
            yc[i,j]=H
            xc[i,j]=xleft+sum(r[nps[i]:j+1])*2+(j-nps[i])*spacing-r[j]
            if xc[i,j]+r[j]>xright:
                nps[i+1]=j
                break
        else:
            nps[i+1]=total_nps
    else:
        nps[i+1]=nps[i]
    
    # boundary-prticle contacts
    for j in range(nps[i+1]):
        ay[i,j]=-g
        delta=yc[i,j]-ybottom-r[j]
        if delta<0:
            ay[i,j]+=-Kn*delta-Cn*vy[i,j]
        delta=xc[i,j]-xleft-r[j]
        if delta<0:
            ax[i,j]+=-Kn*delta-Cn*vx[i,j]
        delta=xright-xc[i,j]-r[j]
        if delta<0:
            ax[i,j]+=Kn*delta-Cn*vx[i,j]
    # particle-particle contacts
    for j in range(nps[i+1]-1):
        for k in range(j+1,nps[i+1]):
            dist=math.sqrt((xc[i,j]-xc[i,k])**2+(yc[i,j]-yc[i,k])**2) #distance between the center of the particles
            delta=dist-r[j]-r[k]
            if delta<0:
                nx=(xc[i,j]-xc[i,k])/dist
                ny=(yc[i,j]-yc[i,k])/dist
                vn=(vx[i,j]-vx[i,k])*nx+(vy[i,j]-vy[i,k])*ny
                Fn=-Kn*delta-Cn*vn 
                ax[i,j]+=Fn*nx/m
                ay[i,j]+=Fn*ny/m
                ax[i,k]+=-Fn*nx/m
                ay[i,k]+=-Fn*ny/m

    # Updating Locations and Velocities
    for j in range(nps[i+1]):
        vx_half[j]+=ax[i,j]*dt
        vy_half[j]+=ay[i,j]*dt
        xc[i+1,j]=xc[i,j]+vx_half[j]*dt
        yc[i+1,j]=yc[i,j]+vy_half[j]*dt
        vx[i+1,j]=vx_half[j]+ax[i,j]*dt/2
        vy[i+1,j]=vy_half[j]+ay[i,j]*dt/2

# Writing the results to a text file
f=open("initial_config.txt","w+")
for j in range(total_nps):
    f.write("%f\t%f\t%f\n" % (r[j],xc[-1,j],yc[-1,j]))
f.close()

# Plotting
nn=32
d_angle=2*math.pi/nn
xcircle=np.zeros(nn)
ycircle=np.zeros(nn)

plt.figure(figsize=(10,10))
# plotting every 0.005sec
for i in range(0,int(T/dt),int(0.01/dt)): 
    plt.clf()
    plt.plot([xleft,xleft,xright,xright],[ytop,ybottom,ybottom,ytop],'k')
    for j in range(nps[i+1]):
        for k in range(nn):
            xcircle[k]=xc[i,j]+math.cos(d_angle*k)*r[j]
            ycircle[k]=yc[i,j]+math.sin(d_angle*k)*r[j]
        plt.fill(xcircle,ycircle,facecolor='#FFA07A',edgecolor='#FF8C00')
    
    plt.axis([-0.5, 1.5, 0, 1.5])
    plt.gca().set_aspect('equal')
    plt.savefig('Frame%07d.png' %i)

