# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math 
import matplotlib.pyplot as plt

# SI Units

# Particle properties
total_nps=4 #total number of particles
m=0.04
r=np.ones(total_nps)*0.01


# Contact properties 
Kn=1000000.
damping=5/100
Ccrit=2*math.sqrt(Kn*m)
Cn=damping*Ccrit
g=9.81

# Geometrical properties
ybottom=-0.05
ytop=0.35
xleft=0.0
xright=0.3
width=xright-xleft

# Discretization
dt=.00001
dx=dy=0.0001
nn=32 # number of nodes on each particle
d_angle=2*math.pi/nn #discretization angle

# Initializing t, x, y, v, omega, and energy arrays
# time
T=.036
t=np.arange(0,T,dt)

# location
xc=np.zeros((len(t),total_nps)) # x of center
yc=np.zeros((len(t),total_nps)) # y of center
xc[0,0]=xc[0,2]=0.142
xc[0,1]=xc[0,3]=0.158
yc[0,0:2]=0.1
yc[0,2:4]=0.05
#f=open("initial_config.txt","r")
#for j in range(total_nps):
#    a=f.readline()
#    b=a.split()
#    r[j]=b[0]
#    xc[0,j]=b[1]
#    yc[0,j]=b[2]

x=np.zeros((len(t),total_nps,nn)) 
y=np.zeros((len(t),total_nps,nn))

for j in range(total_nps):
    for k in range(nn):
        x[0,j,k]=xc[0,j]+r[j]*math.cos(d_angle*k)
        y[0,j,k]=yc[0,j]+r[j]*math.sin(d_angle*k)

# velocity
vx=np.zeros((len(t),total_nps)) #velocity of the center of mass
vy=np.zeros((len(t),total_nps)) 
vx_half=np.zeros(total_nps)
vy_half=np.zeros(total_nps)
omega=np.zeros(total_nps)
omega_half=np.zeros(total_nps)

# acceleration
ax=np.zeros((len(t),total_nps))
ay=np.ones((len(t),total_nps))
alpha=np.zeros((len(t),total_nps))

# boundary nodes
boundary=np.zeros(total_nps)
#for j in range(total_nps):
#    if yc[0,j]<0.02:
#        vy[:,j]=0.
#        boundary[j]=1

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
    # applying periodic boundary (shifting the out of bound coordinates)
    for j in range(total_nps): #jm: master , js: slave
        for k in range(nn):
            if x[i,j,k]<xleft:
                x[i,j,k]+=width
            if x[i,j,k]>xright:
                x[i,j,k]-=width
    for jm in range(total_nps-1): #jm: master , js: slave
        for k in range(nn):
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
                    normal_v=(vx[i,jm]-vx[i,js])*normal_x+(vy[i,jm]-vy[i,js])*normal_y
            
                    # updating the accelerations
                    Fn=-Kn*LSi-Cn*normal_v
                    if jm>1 and Fn<0: #top layer with attractive force, bottom layer (j=2) no attractive force
                        Fn=0
                    # master particle
                    ax[i,jm]+=Fn*normal_x/m
                    ay[i,jm]+=Fn*normal_y/m
                    # slave particle
                    ax[i,js]+=-Fn*normal_x/m
                    ay[i,js]+=-Fn*normal_y/m
                    #print(ax[i,jm],ay[i,jm])
    # Updating Locations and Velocities
    for j in range(total_nps):
        #print(j,boundary[j])
        if boundary[j]==1:
            xc[i+1,j]=xc[i,j]
            yc[i+1,j]=yc[i,j]
        else:
            vx_half[j]+=ax[i,j]*dt
            vy_half[j]+=ay[i,j]*dt
            xc[i+1,j]=xc[i,j]+vx_half[j]*dt
            yc[i+1,j]=yc[i,j]+vy_half[j]*dt
            vx[i+1,j]=vx_half[j]+ax[i,j]*dt/2
            vy[i+1,j]=vy_half[j]+ay[i,j]*dt/2
        for k in range(nn):
            x[i+1,j,k]=xc[i+1,j]+r[j]*math.cos(d_angle*k)
            y[i+1,j,k]=yc[i+1,j]+r[j]*math.sin(d_angle*k)
        
        # finding the indices required for updating LS
        i_delta_x[j]=int(round((xc[i+1,j]-xmid)/dx))
        i_delta_y[j]=int(round((yc[i+1,j]-ymid)/dy))
        #print(xc[i+1,j],xc[0,j],i_delta_x[j],i_delta_y[j])

# Plotting
plt.figure(figsize=(10,5))
# plotting every 0.0005sec
for i in range(0,int(T/dt),int(0.0005/dt)): 
    plt.clf()
    #plt.plot([0.,0.,0.3,0.3],[0.15,0.,0.,0.15],'k')
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
        key=0
        for k in range(nn):
            if abs(x[i,j,k]-x[i,j,k-1])>0.01: 
                if key==0: key=1
                else: key=0
            if key==0:
                x_in.append(x[i,j,k])
                y_in.append(y[i,j,k])
            else:
                x_out.append(x[i,j,k])
                y_out.append(y[i,j,k])
            plt.fill(x_out,y_out,facecolor=fcolor,edgecolor=ecolor)
            plt.fill(x_in,y_in,facecolor=fcolor,edgecolor=ecolor)
        
    
    plt.axis([0., 0.3, 0., 0.2])
    plt.gca().set_aspect('equal')
    plt.savefig('Frame%07d.png' %i)

