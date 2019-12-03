# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math 
import matplotlib.pyplot as plt

# SI Units

# Particle properties
total_nps=100 #total number of particles
m=0.04
r=np.zeros(total_nps)

# Contact properties 
Kn=Kt=1e7
damping=0.05
Ccrit=2*np.sqrt(Kn*m)
Cn=damping*Ccrit
mu=0.5
g=9.81

# Geometrical properties
ybottom=-0.05
ytop=0.25
xright=0.3
xleft=0.
width=xright-xleft

# Discretization
dt=.000001
dtcrit=2*(math.sqrt(1+damping**2)-damping)/math.sqrt(Kn/m)
if dt>0.1*dtcrit:
    print("critical timestep exceeded")
print('the ciritical timestep is {0:f} and the timestep used is {1:f}'.format(dtcrit,dt))

# Initializing t, x, y, v, omega, and energy arrays
# time
T=.01
t=np.arange(0,T,dt)

# location
xc=np.zeros((len(t),total_nps)) # x of center
yc=np.zeros((len(t),total_nps)) # y of center

f=open("initial_config.txt","r")
for j in range(total_nps):
    a=f.readline()
    b=a.split()
    r[j]=b[0]
    xc[0,j]=b[1]
    yc[0,j]=b[2]

I=0.5*m*r**2

# velocity
vx=np.zeros((len(t),total_nps)) #velocity of the center of mass
vy=np.zeros((len(t),total_nps)) 
vx_half=np.zeros(total_nps)
vy_half=np.zeros(total_nps)
omega=np.zeros((len(t),total_nps))
omega_half=np.zeros(total_nps)

# acceleration
ax=np.zeros((len(t),total_nps))
ay=np.ones((len(t),total_nps))*-g
alpha=np.zeros((len(t),total_nps))

# boundary nodes
boundary=np.zeros(total_nps)
v0=10.
for j in range(total_nps):
    if yc[0,j]<0.025:
        vx[:,j]=v0
        boundary[j]=1
    if yc[0,j]>0.14:
        vx[:,j]=-v0
        #vy[:,j]=-v0
        boundary[j]=1

# Updating the Kinematic Properties of Particles
for i in range(len(t)-1):
    print(i)
    for jm in range(total_nps-1): #jm: master , js: slave
            for js in range(jm+1,total_nps):
                lx=xc[i,jm]-xc[i,js] 
                if abs(lx)<(width-r[jm]-r[js]):
                    distance=np.sqrt((xc[i,jm]-xc[i,js])**2+(yc[i,jm]-yc[i,js])**2) 
                else:
                    distance=np.sqrt((xc[i,jm]-xc[i,js]-np.sign(lx)*width)**2+(yc[i,jm]-yc[i,js])**2) 

                delta=distance-r[jm]-r[js]
                if delta<0:
                    # finding the normal and tangent unit vectors
                    if abs(lx)<(width-r[jm]-r[js]):
                        normal_x=(xc[i,jm]-xc[i,js])/distance
                    else:
                        normal_x=(xc[i,jm]-xc[i,js]-np.sign(lx)*width)/distance
                    
                    normal_y=(yc[i,jm]-yc[i,js])/distance
                    tangent_x=-normal_y
                    tangent_y=normal_x
                    normal_v=(vx[i,jm]-vx[i,js])*normal_x+(vy[i,jm]-vy[i,js])*normal_y
                    tangent_v=(vx[i,jm]-vx[i,js])*tangent_x+(vy[i,jm]-vy[i,js])*tangent_y
                    tangent_v+=-r[jm]*omega[i,jm]-r[js]*omega[i,js]
                    # updating the accelerations
                    Fn=-Kn*delta-Cn*normal_v 
                    if Fn<0.:
                        Fn=0.
                    Ft=-Kt*tangent_v*dt
                    if abs(Ft)>=mu*Fn:
                        Ft=np.sign(Ft)*mu*Fn
                    # master particle
                    ax[i,jm]+=(Fn*normal_x+Ft*tangent_x)/m
                    ay[i,jm]+=(Fn*normal_y+Ft*tangent_y)/m
                    alpha[i,jm]+=-Ft/I[jm]
                    # slave particle
                    ax[i,js]+=-(Fn*normal_x+Ft*tangent_x)/m
                    ay[i,js]+=-(Fn*normal_y+Ft*tangent_y)/m
                    alpha[i,js]+=-Ft/I[js]
                    #print(ax[i,jm],ay[i,jm])

    # Updating Locations and Velocities
    for j in range(total_nps):
        #print(j,boundary[j])
        if boundary[j]==1:
            xc[i+1,j]=xc[i,j]+vx[i,j]*dt
            yc[i+1,j]=yc[i,j]
        else:
            vx_half[j]+=ax[i,j]*dt
            vy_half[j]+=ay[i,j]*dt
            omega_half[j]+=alpha[i,j]*dt
            xc[i+1,j]=xc[i,j]+vx_half[j]*dt
            yc[i+1,j]=yc[i,j]+vy_half[j]*dt
            vx[i+1,j]=vx_half[j]+ax[i,j]*dt/2
            vy[i+1,j]=vy_half[j]+ay[i,j]*dt/2
            omega[i+1,j]=omega_half[j]+alpha[i,j]*dt/2

# Plotting
plt.figure(figsize=(10,5))
# plotting every 0.005sec
nn=32 # number of nodes on each particle
d_angle=2*math.pi/nn #discretization angle
for i in range(0,int(T/dt),int(0.0001/dt)): 
    plt.clf()
    plt.plot([0.,0.,0.3,0.3],[0.15,0.,0.,0.15],'k')
    for j in range(total_nps):
        fcolor='#FFA07A'
        ecolor='#FF8C00'
        if boundary[j]==1:
            fcolor='#ADD8E6'
            ecolor='#0000A0'
        x_in=[]
        y_in=[]
        x_out=[]
        y_out=[]
        for k in range(nn):
            x=xc[i,j]+r[j]*math.cos(d_angle*k) 
            y=yc[i,j]+r[j]*math.sin(d_angle*k)
            if x<xleft:
                x+=width
                x_out.append(x)
                y_out.append(y)
            elif x>xright:
                x-=width
                x_out.append(x)
                y_out.append(y)
            else:
                x_in.append(x)
                y_in.append(y)
        plt.fill(x_out,y_out,facecolor=fcolor,edgecolor=ecolor)
        plt.fill(x_in,y_in,facecolor=fcolor,edgecolor=ecolor)
        
    
    plt.axis([-0.05, 0.35, 0, 0.4])
    plt.gca().set_aspect('equal')
    plt.savefig('Frame%07d.png' %i)

