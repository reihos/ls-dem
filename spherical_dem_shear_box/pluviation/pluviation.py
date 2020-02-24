# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math 
import matplotlib.pyplot as plt

# SI Units
g=9.81

# Particle properties
r=[]
f=open('radius_998_particles_Dmax_over_Dmin=10.txt','r')
lines=f.read().splitlines()
f.close()
for line in lines:
    r.append(float(line))
np.random.shuffle(r)
r_avg=np.average(r)
total_nps=len(r)
r=np.array(r)
rho=3000
m=rho*np.pi*r**2

# Contact properties 
Kn=1e5
damping=0.5
Ccrit=2*np.sqrt(Kn*np.average(m))
Cn=damping*Ccrit

# Geometrical properties
ybottom=0.
ytop=1.
xright=1.
xleft=0.

# Discretization
dt=.0001
dtcrit=2*(math.sqrt(1+damping**2)-damping)/math.sqrt(Kn/np.average(m))
print('the ciritical timestep is {0:f} and the timestep used is {1:f}'.format(dtcrit,dt))
#if dt>0.1*dtcrit:
#    raise RuntimeError("critical timestep exceeded")

# Drop properties
tdrop=0.05
idrop=int(tdrop/dt)
spacing=r_avg
H0=0.1 # drop height of the first layer

# Initializing t, x, y, v, a 
# time
T=3
t=np.arange(0,T,dt)

# location
xc=np.zeros(total_nps) # x of center
yc=np.zeros(total_nps) # y of center

# velocity
vx=np.zeros(total_nps) #velocity of the center of mass
vy=np.zeros(total_nps) 
vx_half=np.zeros(total_nps)
vy_half=np.ones(total_nps)*g*dt/2

# acceleration
ax=np.zeros(total_nps)
ay=np.zeros(total_nps)

nps=0

# Updating the Kinematic Properties of Particles
for i in range(len(t)-1):
    print(i) if i%100==0 else None
    if i%idrop==0 and nps<total_nps:
        H=H0+2*r_avg*t[i]/tdrop
        for j in range(nps, total_nps):
            yc[j]=H
            xc[j]=xleft+sum(r[nps:j+1])*2+(j-nps)*spacing-r[j]
            if xc[j]+r[j]>xright:
                nps=j
                break
        else:
            nps=total_nps
    
    # boundary-prticle contacts
    for j in range(nps):
        ay[j]=-g
        ax[j]=0
        delta=yc[j]-ybottom-r[j] #bottom boundary
        if delta<0:
            ay[j]+=(-Kn*delta-Cn*vy[j])/m[j]
        delta=xc[j]-xleft-r[j] #left boundary
        if delta<0:
            ax[j]+=(-Kn*delta-Cn*vx[j])/m[j]
        delta=xright-xc[j]-r[j] #right boundary
        if delta<0:
            ax[j]+=(Kn*delta-Cn*vx[j])/m[j]
    # particle-particle contacts
    for j in range(nps-1):
        for k in range(j+1,nps):
            dist=math.sqrt((xc[j]-xc[k])**2+(yc[j]-yc[k])**2) #distance between the center of the particles
            delta=dist-r[j]-r[k]
            if delta<0:
                nx=(xc[j]-xc[k])/dist
                ny=(yc[j]-yc[k])/dist
                vn=(vx[j]-vx[k])*nx+(vy[j]-vy[k])*ny
                Fn=-Kn*delta-Cn*vn 
                ax[j]+=Fn*nx/m[j]
                ay[j]+=Fn*ny/m[j]
                ax[k]+=-Fn*nx/m[k]
                ay[k]+=-Fn*ny/m[k]

    # Updating Locations and Velocities
    for j in range(nps):
        vx_half[j]+=ax[j]*dt
        vy_half[j]+=ay[j]*dt
        xc[j]+=vx_half[j]*dt
        yc[j]+=vy_half[j]*dt
        vx[j]=vx_half[j]+ax[j]*dt/2
        vy[j]=vy_half[j]+ay[j]*dt/2

    # Plotting
    tplot=0.1
    if i%int(tplot/dt)==0:
        nn=32
        d_angle=2*math.pi/nn
        xcircle=np.zeros(nn)
        ycircle=np.zeros(nn)

        plt.figure(figsize=(10,10))
        plt.plot([xleft,xleft,xright,xright],[ytop,ybottom,ybottom,ytop],'k')
        for j in range(nps):
            for k in range(nn):
                xcircle[k]=xc[j]+math.cos(d_angle*k)*r[j]
                ycircle[k]=yc[j]+math.sin(d_angle*k)*r[j]
            plt.fill(xcircle,ycircle,facecolor='#FFA07A',edgecolor='#FF8C00')
        
        plt.axis([-0.5*xright, 1.5*xright, 0, 1.5*ytop])
        plt.gca().set_aspect('equal')
        plt.savefig('Frame%07d.png' %i)
        plt.close()

# Writing the results to a text file
f=open("initial_config.txt","w+")
for j in range(total_nps):
    f.write("%f\t%f\t%f\n" % (r[j],xc[j],yc[j]))
f.close()
