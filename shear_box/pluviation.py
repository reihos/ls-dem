# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math 
import matplotlib.pyplot as plt

# SI Units

# Particle properties
total_nps=100 #total number of particles
m=0.04
r=np.ones(total_nps)*0.01
r[3]=r[6]=r[10]=r[15]=r[22]=r[24]=r[36]=r[43]=r[48]=r[56]=0.02
#r=np.zeros(total_nps)
#f=open('radius.txt','r')
#for j in range(total_nps):
#    r[j] = float(f.readline())
#np.random.shuffle(r)
r_avg=np.average(r)

# Contact properties 
Kn=100000.
Cn=100.

# Initial Conditions]
H0=0.2
g=9.81

# Geometrical properties
ybottom=0.
xright=0.3
xleft=0.

# Discretization
dt=.00001

# Drop properties
npdrop=10
tdrop=0.1
idrop=int(tdrop/dt)

# Initializing t, x, y, v, omega, and energy arrays
# time
T=1.5
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

nps=0
drop_count=0

# Updating the Kinematic Properties of Particles
for i in range(len(t)-1):
    if i%idrop==0 and nps<total_nps:
        nps+=npdrop
        H=H0+2*r_avg*t[i]/tdrop
        for jdrop in range(npdrop*drop_count,nps):
            print(t[i],jdrop)
            yc[i,jdrop]=H
            xstart=0.02
            if drop_count%2==0:
                xstart=0.01
            xc[i,jdrop]=xstart+(jdrop-npdrop*drop_count)*0.03
        drop_count+=1

    # boundary-prticle contacts
    for j in range(nps):
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
    for j in range(nps-1):
        for k in range(j+1,nps):
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
    for j in range(nps):
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
nps=0
nn=32
d_angle=2*math.pi/nn
xcircle=np.zeros(nn)
ycircle=np.zeros(nn)

plt.figure(figsize=(10,5))
# plotting every 0.005sec
for i in range(0,int(T/dt),int(0.005/dt)): 
    plt.clf()
    plt.plot([0.,0.,0.3,0.3],[0.3,0.,0.,0.3],'k')
    nps=min((math.floor(t[i]/tdrop)+1)*npdrop,total_nps)
    for j in range(nps):
        for k in range(nn):
            xcircle[k]=xc[i,j]+math.cos(d_angle*k)*r[j]
            ycircle[k]=yc[i,j]+math.sin(d_angle*k)*r[j]
        plt.fill(xcircle,ycircle,facecolor='#FFA07A',edgecolor='#FF8C00')
    
    plt.axis([-0.05, 0.35, 0, 0.4])
    plt.gca().set_aspect('equal')
    plt.savefig('Frame%07d.png' %i)

