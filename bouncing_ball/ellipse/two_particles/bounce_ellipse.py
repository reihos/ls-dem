# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math 
import matplotlib.pyplot as plt
import json

# SI Units

# Particle properties
nps=2 #number of particles #particle 0 circle, particle 1 ellipse
m=np.array([0.0654498,0.0654498])
aa=np.array([0.025, 0.05]) #horizontal radius 
bb=np.array([0.025, 0.025]) #vertical radius
I=m/4*(aa**2+bb**2)
theta=[0/180*math.pi, 0/180*math.pi]  #angles of the balls

# Contact properties 
Kn=10000.
#Cn=5.772
Cn=0.

# Initial Conditions
H0=0.2
g=9.81

# Geometrical properties
ywall=0.
ytop=0.5
ybottom=-0.05
xright=0.4
xleft=-0.4

# Discretization
dx=dy=0.001
dt=0.0001
nn=32 #number of boundary nodes for the particle
d_angle=2*math.pi/nn #discretization angle

# Initializing t, x, y, v, omega, and energy arrays
# time
T=1
t=np.arange(0,T,dt)

# location
xc=np.zeros((len(t),nps)) # x of center
yc=np.zeros((len(t),nps)) # y of center
yc[0,0]=0.025
yc[0,1]=H0
#xc[0,0]=-0.05
#xc[0,1]=0.05
x=np.zeros((len(t),nn,nps)) 
y=np.zeros((len(t),nn,nps))

r=np.zeros((nn,nps))
for j in range(nn):
    for k in range(nps):
        r[j,k]= aa[k]*bb[k]/math.sqrt((bb[k]*math.cos(d_angle*j))**2+(aa[k]*math.sin(d_angle*j))**2)
        x[0,j,k]=xc[0,k]+r[j,k]*math.cos(d_angle*(j)+theta[k])
        y[0,j,k]=yc[0,k]+r[j,k]*math.sin(d_angle*(j)+theta[k])

# velocity
vx=np.zeros((len(t),nps)) #velocity of the center of mass
vy=np.zeros((len(t),nps)) 
vnx=np.zeros((len(t),nn,nps)) #velocity of each node on the particle
vny=np.zeros((len(t),nn,nps))
vx_half=np.zeros(nps)
vy_half=np.ones(nps)*g*dt/2
vnx_half=np.zeros((nn,nps))
vny_half=np.zeros((nn,nps))
omega=np.zeros(nps)
omega_half=np.zeros(nps)

# acceleration
ax=np.zeros((len(t),nps))
ay=np.zeros((len(t),nps))
alpha=np.zeros((len(t),nps))

# energy
Ep=np.zeros((len(t)-1,nps))
Ek=np.zeros((len(t)-1,nps))
Et=np.zeros((len(t)-1,nps))

# Defining the level set functions
ny=int((ytop-ybottom)/dy)
nx=int((xright-xleft)/dx)
# for the wall
LSw=np.zeros(ny)
for yy in range(ny):
    LSw[yy]=ybottom+yy*dy
# for particle 0
LSp=np.zeros((nx,ny))
for xx in range(nx):
    for yy in range(ny):
        xg=xleft+xx*dx # x of grid point
        yg=ybottom+yy*dy
        R=math.sqrt((xg-xc[0,0])**2+(yg-yc[0,0])**2) # distance from grid point to center of the particle
        LSp[xx,yy]=R-aa[0]

ibottom=ybottom/dy #needed in the next step
ileft=xleft/dx
LSi=np.zeros(nn)
i_delta_x=0
i_delta_y=0

# Updating the Kinematic Properties of Particles
for i in range(len(t)-1):
    # Updating the Accelerations
    # contact between all particles and the wall
    for k in range(nps):
        ay[i,k]=-g   
        for j in range(nn):
            # finding the index of the node 
            yf=math.floor(y[i,j,k]/dy)*dy
            iyf=int(math.floor(y[i,j,k]/dy)-ibottom)
            # interpolating
            LSi[j]=LSw[iyf]+(y[i,j,k]-yf)*(LSw[iyf+1]-LSw[iyf])/dy
            
            # updating the accelerations
            if LSi[j]<0: 
                ay[i,k]+=(-Kn*LSi[j]-Cn*vny[i,j,k])/m[k] #linear acceleration in the y direction
                alpha[i,k]+=(-Kn*LSi[j]-Cn*vny[i,j,k])*(x[i,j,k]-xc[i,k])/I[k] #angular acceleration
                Ep[i,k]+=0.5*Kn*LSi[j]**2
    
    # contact between the two particles
    for j in range(nn):
        # finding the indices of the node
        xf=math.floor(x[i,j,1]/dx)*dx
        yf=math.floor(y[i,j,1]/dy)*dy
        ixf=int(math.floor(x[i,j,1]/dx)-ileft)
        iyf=int(math.floor(y[i,j,1]/dy)-ibottom)
        # updating LSp indices based on the spheres new location
        ixf=ixf-i_delta_x
        iyf=iyf-i_delta_y
        # interpolating
        LSi[j]=LSp[ixf,iyf]*(xf+dx-x[i,j,1])*(yf+dy-y[i,j,1])+LSp[ixf+1,iyf]*(x[i,j,1]-xf)*(yf+dy-y[i,j,1])+LSp[ixf,iyf+1]*(xf+dx-x[i,j,1])*(y[i,j,1]-yf)+LSp[ixf+1,iyf+1]*(x[i,j,1]-xf)*(y[i,j,1]-yf)
        LSi[j]=LSi[j]/dx/dy

        if LSi[j]<0:
            # finding the normal direction
            dLS_dx=((yf+dy-y[i,j,1])*(LSp[ixf+1,iyf]-LSp[ixf,iyf])+(y[i,j,1]-yf)*(LSp[ixf+1,iyf+1]-LSp[ixf,iyf+1]))/dx/dy
            dLS_dy=((xf+dx-x[i,j,1])*(LSp[ixf,iyf+1]-LSp[ixf,iyf])+(x[i,j,1]-xf)*(LSp[ixf+1,iyf+1]-LSp[ixf+1,iyf]))/dx/dy
            mag_dLS=math.sqrt(dLS_dx**2+dLS_dy**2)
            normal_x=dLS_dx/mag_dLS
            normal_y=dLS_dy/mag_dLS
            normal_v=(vnx[i,j,1]-vx[i,0])*normal_x+(vny[i,j,1]-vy[i,0])*normal_y
            
            # updating the accelerations
            # slave particle
            Fn=-Kn*LSi[j]-Cn*normal_v 
            ax[i,1]+=Fn*normal_x/m[k]
            ay[i,1]+=Fn*normal_y/m[k]
            alpha[i,1]+=Fn*(normal_x*(y[i,j,1]-yc[i,1])+normal_y*(x[i,j,1]-xc[i,1]))/I[k]
            # master particle
            ax[i,0]+=-Fn*normal_x/m[k]
            ay[i,0]+=-Fn*normal_y/m[k]
            alpha[i,0]+=-Fn*(normal_x*(y[i,j,0]-yc[i,0])+normal_y*(x[i,j,0]-xc[i,0]))/I[k]

            Ep[i,:]+=0.5*Kn*LSi[j]**2

    # Updating Locations and Velocities
    for k in range(nps):
        # updating the location of the center of the particles
        #xc[i+1,k]=xc[i,k]+vx[i,k]*dt+0.5*ax[i,k]*dt**2
        #yc[i+1,k]=yc[i,k]+vy[i,k]*dt+0.5*ay[i,k]*dt**2
        vx_half[k]+=ax[i,k]*dt
        vy_half[k]+=ay[i,k]*dt
        xc[i+1,k]=xc[i,k]+vx_half[k]*dt
        yc[i+1,k]=yc[i,k]+vy_half[k]*dt
        # calculating the rotation angle
        #theta[k]+=omega[k]*dt+0.5*alpha[i,k]*dt**2
        omega_half[k]+=alpha[i,k]*dt
        theta[k]+=omega_half[k]*dt
        # updating the location of all the nodes on the particle
        for j in range(nn):
            x[i+1,j,k]=xc[i+1,k]+r[j,k]*math.cos(d_angle*j+theta[k])
            y[i+1,j,k]=yc[i+1,k]+r[j,k]*math.sin(d_angle*j+theta[k])
        
        # updating the velocities
        #vx[i+1,k]=vx[i,k]+ax[i,k]*dt
        #vy[i+1,k]=vy[i,k]+ay[i,k]*dt
        #omega[k]+=alpha[i,k]*dt
        vx[i+1,k]=vx_half[k]+ax[i,k]*dt/2
        vy[i+1,k]=vy_half[k]+ay[i,k]*dt/2
        omega[k]=omega_half[k]+alpha[i,k]*dt/2
        for j in range(nn):
            #vnx[i+1,j,k]=vnx[i,j,k]+ax[i,k]*dt+alpha[i,k]*dt*r[j,k]*-math.sin(d_angle*j+theta[k])
            #vny[i+1,j,k]=vny[i,j,k]+ay[i,k]*dt+alpha[i,k]*dt*r[j,k]*math.cos(d_angle*j+theta[k])
            vnx_half[j,k]+=ax[i,k]*dt+alpha[i,k]*dt*r[j,k]*-math.sin(d_angle*j+theta[k])  
            vny_half[j,k]+=ay[i,k]*dt+alpha[i,k]*dt*r[j,k]*math.cos(d_angle*j+theta[k])  
            vnx[i+1,j,k]=vnx_half[j,k]+ax[i,k]*dt/2+alpha[i,k]*dt/2*r[j,k]*-math.sin(d_angle*j+theta[k])
            vny[i+1,j,k]=vny_half[j,k]+ay[i,k]*dt/2+alpha[i,k]*dt/2*r[j,k]*math.cos(d_angle*j+theta[k])

        # caculating the energies
        Ep[i,k]+=m[k]*g*yc[i,k]
        Ek[i,k]=0.5*m[k]*(vx[i,k]**2+vy[i,k]**2)+0.5*I[k]*omega[k]**2
        Et[i,k]=Ep[i,k]+Ek[i,k]

    # finding the indices required for updating LSp
    i_delta_x=int(round((xc[i+1,0]-xc[0,0])/dx))
    i_delta_y=int(round((yc[i+1,0]-yc[0,0])/dy))
   
   # for xx in range(nx):
   #     for yy in range(ny):
   #         xg=xleft+xx*dx 
   #         yg=ybottom+yy*dy
   #         R=math.sqrt((xg-xc[i+1,0])**2+(yg-yc[i+1,0])**2)
   #         LSp[xx,yy]=R-aa[0]


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
with open("lammps.json","r") as f:
    data=json.load(f)

# Plotting
# plot 1: height vs time  plot 2: energy vs time  
# plot 3: 2D location   plot 4: acceleration and velocity vs time
plot='2'
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
    plt.plot(t[:-1],Et[:,0]+Et[:,1],label="total energy")
    plt.xlabel("time (sec)")
    plt.ylabel("energy (J)")
    plt.legend()
    plt.show()

if '3' in plot:
    plt.figure(figsize=(10,5))
    fcolor=['#FFA07A','#A52A2A']
    ecolor=['#FF8C00','#D2691E']  
    # plotting every 0.005sec
    for n in range(0,int(T/dt),int(0.005/dt)): 
        plt.cla()
        for k in range(nps):
            plt.fill(np.append(x[n,:,k],x[n,0,k]),np.append(y[n,:,k],y[n,0,k]),facecolor=fcolor[k],edgecolor=ecolor[k])
        plt.axis([-0.3, 0.3, 0, 0.3])
        plt.gca().set_aspect('equal')
        plt.savefig('Frame%07d.png' %n)

if '4' in plot:
    fig, axs=plt.subplots(3)
    axs[0].plot(t,ay,'k')
    axs[0].set(ylabel=r"$a\ (m/s^2)$")
    axs[1].plot(t,alpha,'k')
    axs[1].set(ylabel=r"$\alpha\ (rad/s^2)$")
    axs[2].plot(t,vy,'k')
    axs[2].set(ylabel=r"$v\ (m/s)$")
    plt.xlabel("time (sec)")
    #xlim=[0.18,0.22]
    #for ax in axs:
    #    ax.set_xlim(xlim)
    plt.show()
