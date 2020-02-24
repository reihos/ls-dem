# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math
import matplotlib.pyplot as plt

# SI Units

# Particle properties
m=0.04
r=0.01
I=0.5*m*r**2

# Contact properties 
Kn=1e5
Kt=1e7
damping=0.05
Ccrit=2*np.sqrt(Kn*m)
Cn=damping*Ccrit
mu=np.tan(np.radians(75))
#mu=0.1
murf=0.
g=9.81

# Geometrical properties
beta=45  #slope angle
beta=beta/180*math.pi
x0=0.5
y0=x0*np.tan(beta)+r/np.cos(beta)

# Discretization
dt=.0001
dtcrit=2*(math.sqrt(1+damping**2)-damping)/math.sqrt(Kn/m)
if dt>0.1*dtcrit:
    print("critical timestep exceeded")
print('the ciritical timestep is {0:f} and the timestep used is {1:f}'.format(dtcrit,dt))

# Initializing t, x, y, v, omega, and energy arrays
# time
T=0.01
t=np.arange(0,T,dt)

# location
xc=np.zeros(len(t)) # x of center
yc=np.zeros(len(t)) # y of center
xc[0]=x0
yc[0]=y0
theta=np.zeros(len(t))

# velocity
vx=np.zeros(len(t)) #velocity of the center of mass
vy=np.zeros(len(t)) 
vx_half=0.
vy_half=0.
omega=np.zeros(len(t))
omega_half=0.

# acceleration
ax=np.zeros(len(t))
ay=np.ones(len(t))*-g
alpha=np.zeros(len(t))

# Updating the Kinematic Properties of Particles
for i in range(len(t)-1):
    print(i)
    delta_slope=abs(-np.tan(beta)*xc[i]+yc[i])*np.cos(beta)-r
    if delta_slope<0:
        normal_x=-np.sin(beta)
        normal_y=np.cos(beta)
        tangent_x=-normal_y
        tangent_y=normal_x
        normal_v=(vx[i])*normal_x+(vy[i])*normal_y
        tangent_v=(vx[i])*tangent_x+(vy[i])*tangent_y
        tangent_v+=-r*omega[i]
        theta_rel=(omega[i])*dt
        
        # caluclating forces and updating the accelerations
        Fn=-Kn*delta_slope-Cn*normal_v 
        if Fn<0.:
            Fn=0.
        Ft=-Kt*tangent_v*dt
        #print(tangent_v,normal_v,omega[i])
        if abs(Ft)>=mu*Fn:
            Ft=np.sign(Ft)*mu*Fn
        Mr=-Kt*r**2*theta_rel
        if abs(Mr)>=murf*Fn*r:
            Mr=np.sign(Mr)*murf*Fn*r
        Fx=Fn*normal_x+Ft*tangent_x 
        Fy=Fn*normal_y+Ft*tangent_y 
        # master particle
        ax[i]+=Fx/m
        ay[i]+=Fy/m
        alpha[i]+=(-Ft*r+Mr)/I
        print("Ft = ",Ft,"Fn = ",Fn)
        print("Normalized Ft = ",Ft/(m*g*np.sin(beta)),"Normalized Fn = ",Fn/(m*g*np.cos(beta)))
    # Updating Locations and Velocities
    vx_half+=ax[i]*dt
    vy_half+=ay[i]*dt
    omega_half+=alpha[i]*dt
    xc[i+1]=xc[i]+vx_half*dt
    yc[i+1]=yc[i]+vy_half*dt
    vx[i+1]=vx_half+ax[i]*dt/2
    vy[i+1]=vy_half+ay[i]*dt/2
    theta[i+1]=theta[i]+omega_half*dt
    omega[i+1]=omega_half+alpha[i]*dt/2
    #print("xc = ",xc[i+1],"theta = ",theta[i+1])

print("x_end = ",xc[-1],"theta_end = ",theta[-1])

#plt.figure(figsize=(10,5))
# plotting every 0.01 sec
nn=32 # number of nodes on each particle
d_angle=2*math.pi/nn #discretization angle
for i in range(0,int(T/dt),int(0.01/dt)): 
    plt.clf()
    plt.plot([0,1.1*x0],[0,1.1*x0*np.tan(beta)],'k')
    fcolor='#FFA07A'
    ecolor='#FF8C00'
    x_in=[]
    y_in=[]
    for k in range(nn):
        x=xc[i]+r*math.cos(d_angle*k) 
        y=yc[i]+r*math.sin(d_angle*k)
        x_in.append(x)
        y_in.append(y)
    plt.fill(x_in,y_in,facecolor=fcolor,edgecolor=ecolor)
    
    x_dot=xc[i]+r/2*math.cos(theta[i])
    y_dot=yc[i]+r/2*math.sin(theta[i])
    plt.plot(x_dot,y_dot,'ok',markersize=.5)   
    
    plt.axis([0,1.1*x0,0,x0*np.tan(beta)*2])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect('equal')
    plt.gca().set_aspect('equal')
    plt.savefig('Frame%07d.png' %i)
