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
Ccrit=2*math.sqrt(Kn*m)
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
dx=dy=0.001
nn=32 # number of nodes on each particle
d_angle=2*math.pi/nn #discretization angle

# Initializing t, x, y, v, omega, and energy arrays
# time
T=.0005
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

x=np.zeros((len(t),total_nps,nn)) 
y=np.zeros((len(t),total_nps,nn))

for j in range(total_nps):
    for k in range(nn):
        x[0,j,k]=xc[0,j]+r[j]*math.cos(d_angle*k)
        y[0,j,k]=yc[0,j]+r[j]*math.sin(d_angle*k)

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
v0=.1
for j in range(total_nps):
    if yc[0,j]<0.025:
        vx[:,j]=v0
        boundary[j]=1
    if yc[0,j]>0.14:
        vx[:,j]=-v0
        #vy[:,j]=-v0
        boundary[j]=1

# Defining the level set functions
ny=int(round((ytop-ybottom)/dy))
nx=int(round((xright-xleft)/dx))
xmid=(xleft+xright)/2
ymid=(ybottom+ytop)/2
LS=np.zeros((nx+1,ny+1,total_nps))
for xx in range(nx+1):
    for yy in range(ny+1):
        xg=xleft+xx*dx # x of grid point
        yg=ybottom+yy*dy
        for j in range(total_nps):
            R=math.sqrt((xg-xmid)**2+(yg-ymid)**2) # distance from grid point to center of the particle
            LS[xx,yy,j]=R-r[0]
ibottom=ybottom/dy #needed in the next step
ileft=xleft/dx
LSi=0.
i_delta_x=np.zeros(total_nps,int)
i_delta_y=np.zeros(total_nps,int)
for j in range(total_nps):
    i_delta_x[j]=int(round((xc[0,j]-xmid)/dx))
    i_delta_y[j]=int(round((yc[0,j]-ymid)/dy))

# Updating the Kinematic Properties of Particles
for i in range(len(t)-1):
    print(i)
    for jm in range(total_nps-1): #jm: master , js: slave
        for k in range(nn):
            if x[i,jm,k]<xleft:
                x[i,jm,k]+=width
            if x[i,jm,k]>xright:
                x[i,jm,k]-=width
            xf=math.floor(x[i,jm,k]/dx)*dx
            yf=math.floor(y[i,jm,k]/dy)*dy
            # absolute indices of a the floor grid point for a node on a master particle 
            ixfm=int(math.floor(x[i,jm,k]/dx)-ileft)
            iyfm=int(math.floor(y[i,jm,k]/dy)-ibottom)
            #print(i,jm,k)
            #print(i,x[i,jm,k],y[i,jm,k],ixfm,iyfm,xf,yf) 
            for js in range(jm+1,total_nps):
                #print(i,jm,js,k)
                #print(js,i_delta_x[js],i_delta_y[js],xc[i,js])
                # updating LS indices
                # relative indices with respect to a particle positioned in the middle of the domain
                ixf=ixfm-i_delta_x[js]
                iyf=iyfm-i_delta_y[js]
                if ixf<0:
                    ixf+=nx
                if ixf>=nx:
                    ixf-=nx
                #print(i_delta_x[js],i_delta_y[js],ixf,iyf)
                # interpolating
                LSi=LS[ixf,iyf,js]*(xf+dx-x[i,jm,k])*(yf+dy-y[i,jm,k])+LS[ixf+1,iyf,js]*(x[i,jm,k]-xf)*(yf+dy-y[i,jm,k])+LS[ixf,iyf+1,js]*(xf+dx-x[i,jm,k])*(y[i,jm,k]-yf)+LS[ixf+1,iyf+1,js]*(x[i,jm,k]-xf)*(y[i,jm,k]-yf)
                LSi=LSi/dx/dy
                #print(LS[ixf,iyf,js],LS[ixf+1,iyf,js],LS[ixf,iyf+1,js],LS[ixf+1,iyf+1,js],LSi)
                if LSi<0:
                    #print('overlap')
                    # finding the normal direction
                    dLS_dx=((yf+dy-y[i,jm,k])*(LS[ixf+1,iyf,js]-LS[ixf,iyf,js])+(y[i,jm,k]-yf)*(LS[ixf+1,iyf+1,js]-LS[ixf,iyf+1,js]))/dx/dy
                    dLS_dy=((xf+dx-x[i,jm,k])*(LS[ixf,iyf+1,js]-LS[ixf,iyf,js])+(x[i,jm,k]-xf)*(LS[ixf+1,iyf+1,js]-LS[ixf+1,iyf,js]))/dx/dy
                    mag_dLS=math.sqrt(dLS_dx**2+dLS_dy**2)
                    normal_x=dLS_dx/mag_dLS
                    normal_y=dLS_dy/mag_dLS
                    tangent_x=-normal_y
                    tangent_y=normal_x
                    normal_v=(vx[i,jm]-vx[i,js])*normal_x+(vy[i,jm]-vy[i,js])*normal_y
                    tangent_v=(vx[i,jm]-vx[i,js])*tangent_x+(vy[i,jm]-vy[i,js])*tangent_y
                    tangent_v+=-r[jm]*omega[i,jm]-r[js]*omega[i,js]
                    # updating the accelerations
                    Fn=-Kn*LSi-Cn*normal_v 
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
        for k in range(nn):
            x[i+1,j,k]=xc[i+1,j]+r[j]*math.cos(d_angle*k)
            y[i+1,j,k]=yc[i+1,j]+r[j]*math.sin(d_angle*k)
        
        # finding the indices required for updating LS
        i_delta_x[j]=int(round((xc[i+1,j]-xmid)/dx))
        i_delta_y[j]=int(round((yc[i+1,j]-ymid)/dy))
        #print(xc[i+1,j],xc[0,j],i_delta_x[j],i_delta_y[j])

# Plotting
plt.figure(figsize=(10,5))
# plotting every 0.005sec
for i in range(0,int(T/dt),int(0.000001/dt)): 
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
            if abs(x[i,j,k]-x[i,j,0])>0.05:
                x_out.append(x[i,j,k])
                y_out.append(y[i,j,k])
            else:
                x_in.append(x[i,j,k])
                y_in.append(y[i,j,k])
            plt.fill(x_out,y_out,facecolor=fcolor,edgecolor=ecolor)
            plt.fill(x_in,y_in,facecolor=fcolor,edgecolor=ecolor)
        
    
    plt.axis([-0.05, 0.35, 0, 0.4])
    plt.gca().set_aspect('equal')
    plt.savefig('Frame%07d.png' %i)

