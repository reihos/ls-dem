# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# SI Units

# Particle properties
total_nps=100 #total number of particles
m=0.04
r=np.zeros(total_nps)

# Contact properties 
Kn=1e5
Kt=0.
damping=0.05
Ccrit=2*np.sqrt(Kn*m)
Cn=damping*Ccrit
mu=0.5
g=9.81

# Geometrical properties
ybottom=0.
ytop=0.165
xright=0.3
xleft=0.
width=xright-xleft

# Window for Stress Calculation
stress_window_top=0.8*(ytop-ybottom)+ybottom
stress_window_bottom=0.2*(ytop-ybottom)+ybottom
stress_window_right=0.8*(xright-xleft)+xleft
stress_window_left=0.2*(xright-xleft)+xleft
window_area=(stress_window_top-stress_window_bottom)*(stress_window_right-stress_window_left)


# Discretization
dt=.0001
dtcrit=2*(math.sqrt(1+damping**2)-damping)/math.sqrt(Kn/m)
if dt>0.1*dtcrit:
    print("critical timestep exceeded")
print('the ciritical timestep is {0:f} and the timestep used is {1:f}'.format(dtcrit,dt))

# Initializing t, x, y, v, omega, and energy arrays
# time
T=1.
t=np.arange(0,T,dt)
T_pressure=0.2

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

# adding roller boundary particles
n_roller=10
r_roller=np.ones(n_roller)*0.01
for j in range(n_roller):
    if j%2!=0:
        r_roller[j]=0.02

x_roller=np.zeros((len(t),n_roller))
y_roller_top=np.zeros((len(t),n_roller))
y_roller_bottom=np.zeros((len(t),n_roller))
for j in range(n_roller):
    x_roller[0,j]=sum(r_roller[:j+1]*2)-r_roller[j]
y_roller_top[0,:]=ytop+max(r_roller)
y_roller_bottom[0,:]=ybottom-max(r_roller)
xc=np.concatenate((xc,x_roller,x_roller),axis=1)
yc=np.concatenate((yc,y_roller_top,y_roller_bottom),axis=1)
r=np.concatenate((r,r_roller,r_roller))
total_nps+=2*n_roller

I=0.5*m*r**2
theta=np.zeros((len(t),total_nps))
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

# Initializing stress arrays
stress_xx=np.zeros(len(t)-1)
stress_yy=np.zeros(len(t)-1)
stress_xy=np.zeros(len(t)-1)
stress_yx=np.zeros(len(t)-1)
inside_window=np.zeros((len(t)-1,total_nps),bool)

# boundary nodes
boundary=np.zeros(total_nps)
v0=.1
P=1000
#top_particles=pd.DataFrame({'index':[],'x':[],'y':[]})
for j in range(total_nps):
    if yc[0,j]<ybottom:
        vx[int(T_pressure/dt):,j]=v0
        boundary[j]=1
    if yc[0,j]>ytop:
        vx[int(T_pressure/dt):,j]=-v0
        #vy[:,j]=-v0
        boundary[j]=2
#        top_particles=top_particles.append(pd.DataFrame({'index':[j],'x':[xc[0,j]],'y':[yc[0,j]]}),ignore_index=True)
#top_particles.sort_values("x",inplace=True)

#boundary conditions for plate
plate=False
if plate:
    vytop=0.
    vybottom=0.
    vxtop=-10.
    vxbottom=10.
    aytop=0.
    vytop_half=0.


# Updating the Kinematic Properties of Particles
for i in range(len(t)-1):
    print(i)
    Fy_roller=0.
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
                    Fx=Fn*normal_x+Ft*tangent_x 
                    Fy=Fn*normal_y+Ft*tangent_y 
                    # master particle
                    ax[i,jm]+=Fx/m
                    ay[i,jm]+=Fy/m
                    alpha[i,jm]+=-Ft/I[jm]
                    # slave particle
                    ax[i,js]+=-Fx/m
                    ay[i,js]+=-Fy/m
                    alpha[i,js]+=-Ft/I[js]
                    if boundary[jm]==2:
                        Fy_roller+=Fy
                    if boundary[js]==2:
                        Fy_roller+=-Fy
                    
                    branch=[xc[i,js]-xc[i,jm],yc[i,js]-yc[i,jm]]
                    if xc[i,jm]<stress_window_right and xc[i,jm]>stress_window_left and yc[i,jm]<stress_window_top and yc[i,jm]>stress_window_bottom:
                        inside_window[i,jm]=True
                        
                        if xc[i,js]<stress_window_right and xc[i,js]>stress_window_left and yc[i,js]<stress_window_top and yc[i,js]>stress_window_bottom:
                            inside_window[i,js]=True
                            stress_xx[i]+=Fx*branch[0]
                            stress_yy[i]+=Fy*branch[1]
                            stress_xy[i]+=Fy*branch[0]
                            stress_yx[i]+=Fx*branch[1]
                        else:
                            scale=distance*r[jm]
                            stress_xx[i]+=Fx*branch[0]*scale
                            stress_yy[i]+=Fy*branch[1]*scale
                            stress_xy[i]+=Fy*branch[0]*scale
                            stress_yx[i]+=Fx*branch[1]*scale

                    elif xc[i,js]<stress_window_right and xc[i,js]>stress_window_left and yc[i,js]<stress_window_top and yc[i,js]>stress_window_bottom:
                        inside_window[i,js]=True
                        scale=distance*r[js]
                        stress_xx[i]+=Fx*branch[0]*scale
                        stress_yy[i]+=Fy*branch[1]*scale
                        stress_xy[i]+=Fy*branch[0]*scale
                        stress_yx[i]+=Fx*branch[1]*scale

                            

    # calculating the stresses in the stress window
    stress_xx[i]/=window_area
    stress_yy[i]/=window_area
    stress_xy[i]/=window_area
    stress_yx[i]/=window_area
    
    if plate:
        for j in range(total_nps):
            # bottom boundary
            delta=abs(yc[i,j]-ybottom)-r[j]
            if delta<0:
                normal_y=1.
                tangent_x=-normal_y
                Fn=-Kn*delta-Cn*(vy[i,j]-vybottom)*normal_y
                if Fn<0.:
                    Fn=0.
                Ft=-Kt*((vx[i,j]-vxbottom)*tangent_x-r[j]*omega[i,j])*dt
                if abs(Ft)>=mu*Fn:
                    Ft=np.sign(Ft)*mu*Fn
            # top boundary
            delta=abs(yc[i,j]-ytop)-r[j]
            if delta<0:
                normal_y=-1.
                tangent_x=-normal_y
                Fn=-Kn*delta-Cn*(vy[i,j]-vytop)*normal_y
                if Fn<0.:
                    Fn=0.
                Ft=-Kt*((vx[i,j]-vxtop)*tangent_x-r[j]*omega[i,j])*dt
                if abs(Ft)>=mu*Fn:
                    Ft=np.sign(Ft)*mu*Fn
            ax[i,j]+=(Ft*tangent_x)/m
            ay[i,j]+=(Fn*normal_y)/m
            alpha[i,j]+=-Ft/I[j]
            #aytop+=-Fn*normal_y/mtop
            
    # Updating Locations and Velocities
    Fy_roller+=-P*2*sum(r_roller)
    ay_roller=Fy_roller/(m*n_roller)
    for j in range(total_nps):
        #print(j,boundary[j])
        if boundary[j]==2:
            xc[i+1,j]=xc[i,j]+vx[i,j]*dt
            ay[i,j]=ay_roller    
            vy_half[j]+=ay[i,j]*dt
            yc[i+1,j]=yc[i,j]+vy_half[j]*dt
            vy[i+1,j]=vy_half[j]+ay[i,j]*dt/2
        elif boundary[j]==1:
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
            theta[i+1,j]=theta[i,j]+omega_half[j]*dt
            omega[i+1,j]=omega_half[j]+alpha[i,j]*dt/2
        #vytop_half+=aytop*dt
        #ytop+=vytop_halp*dt
        #vytop=vytop_half+aytop*dt/2

# Plotting
plt.figure(1,figsize=(10,5))
plt.plot(t[:-1],-stress_xx,'k',label='$\sigma_{xx}$')
plt.plot(t[:-1],-stress_yy,'r',label='$\sigma_{yy}$')
plt.plot(t[:-1],-stress_xy,'b',label='$\sigma_{xy}$')
plt.plot(t[:-1],-stress_yx,'g',label='$\sigma_{yx}$')
plt.gca().ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.legend()
plt.xlabel('time (sec)')
plt.ylabel('Stress (N/m)')
plt.savefig('stress.png')

plt.figure(figsize=(10,5))
# plotting every 0.01 sec
nn=32 # number of nodes on each particle
d_angle=2*math.pi/nn #discretization angle
for i in range(0,int(T/dt),int(0.01/dt)): 
    plt.clf()
    plt.plot([xleft,xleft],[ytop,ybottom],'k')
    plt.plot([xright,xright],[ytop,ybottom],'k')
    plt.plot([stress_window_left,stress_window_right,stress_window_right,stress_window_left,stress_window_left],[stress_window_bottom,stress_window_bottom,stress_window_top,stress_window_top,stress_window_bottom],'b')
    if plate:
        plt.plot([xright,xleft],[ytop,ytop],'b',linewidth=5)
        plt.plot([xright,xleft],[ybottom,ybottom],'b',linewidth=5)
    for j in range(total_nps):
        fcolor='#FFA07A'
        ecolor='#FF8C00'
        if boundary[j]!=0:
            fcolor='#ADD8E6'
            ecolor='#0000A0'
        if inside_window[i,j]:
            fcolor='#90EE90'
            ecolor='#5A8B2A'
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
        
        x_dot=xc[i,j]+r[j]/2*math.cos(theta[i,j])
        y_dot=yc[i,j]+r[j]/2*math.sin(theta[i,j])
        if x_dot<xleft: x_dot+=width
        if x_dot>xright: x_dot-=width
        plt.plot(x_dot,y_dot,'ok',markersize=2)   
    
    plt.axis([-0.05, 0.35, -0.02, 0.4])
    plt.gca().set_aspect('equal')
    plt.savefig('Frame%07d.png' %i)
