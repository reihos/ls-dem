# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# SI Units
g=9.81

# Particle properties and Initial configuration
rho=3000
f=open("initial_config_998.txt","r")
#f=open("config_after_pressure=2000.txt","r")
lines=f.read().splitlines()
total_nps=len(lines)
r=np.zeros(total_nps)
xc=np.zeros(total_nps) # x of center
yc=np.zeros(total_nps) # y of center
j=0
for line in lines:
    s=line.split()
    r[j]=s[0]
    xc[j]=s[1]
    yc[j]=s[2]
    j+=1
print('number of particles =',total_nps)

# Geometrical properties
ybottom=0.
ytop=.5
xright=1.
xleft=0.
width=xright-xleft

# adding roller boundary particles
n_roller=33
r_roller=np.ones(n_roller)*0.01
for j in range(n_roller):
    if j%2==0:
        r_roller[j]=0.02
x_roller=np.zeros(n_roller)
for j in range(n_roller):
    x_roller[j]=sum(r_roller[:j+1]*2)-r_roller[j]
y_roller_top=np.zeros(n_roller)
y_roller_bottom=np.zeros(n_roller)
y_roller_top[:]=0.545
y_roller_bottom[:]=ybottom-max(r_roller)
xc=np.concatenate((xc,x_roller,x_roller))
yc=np.concatenate((yc,y_roller_top,y_roller_bottom))
r=np.concatenate((r,r_roller,r_roller))
jtop=total_nps
jbottom=total_nps+n_roller
total_nps+=2*n_roller

m=rho*np.pi*r**2
I=0.5*m*r**2
m_roller=(m[jtop]+m[jtop+1])/2
theta=np.zeros(total_nps)

# Contact properties 
Kn=1e5
Kt=1e5
damping=0.05
Ccrit=2*np.sqrt(Kn*np.average(m))
Cn=damping*Ccrit
mu=0.5
murf=0.5

# Window for Stress Calculation
stress_window_top=0.8*(ytop-ybottom)+ybottom
stress_window_bottom=0.2*(ytop-ybottom)+ybottom
stress_window_right=0.8*(xright-xleft)+xleft
stress_window_left=0.2*(xright-xleft)+xleft
window_area=(stress_window_top-stress_window_bottom)*(stress_window_right-stress_window_left)

# Discretization
dt=.0001
dtcrit=2*(math.sqrt(1+damping**2)-damping)/math.sqrt(Kn/np.average(m))
print('the ciritical timestep is {0:f} and the timestep used is {1:f}'.format(dtcrit,dt))
#if dt>0.1*dtcrit:
#    raise RuntimeError("critical timestep exceeded")

# Initializing arrays

# time
T=1.5
t=np.arange(0,T,dt)
T_pressure=T
uniaxial_compression=True

# velocity
vx=np.zeros(total_nps) #velocity of the center of mass
vy=np.zeros(total_nps) 
vx_half=np.zeros(total_nps)
vy_half=np.zeros(total_nps)
omega=np.zeros(total_nps)
omega_half=np.zeros(total_nps)

# acceleration
ax=np.zeros(total_nps)
ay=np.zeros(total_nps)
alpha=np.zeros(total_nps)

# Initializing stress arrays
stress_xx=np.zeros(len(t))
stress_yy=np.zeros(len(t))
stress_xy=np.zeros(len(t))
stress_yx=np.zeros(len(t))
inside_window=np.zeros(total_nps,bool)

Normal_Force=np.zeros((total_nps,total_nps))
contact=np.zeros((total_nps,total_nps),bool)
# boundary nodes
boundary=np.zeros(total_nps)
v0=.1
P=0
for j in range(total_nps):
    if j>=jbottom: #bottom
        boundary[j]=1
    elif j>=jtop: #top
        boundary[j]=2

# Updating the Kinematic Properties of Particles
for i in range(len(t)):
    print(i)
    # reseting values and arrays to zero
    Fy_roller=0.
    ax.fill(0)
    ay.fill(-g)
    alpha.fill(0)
    Normal_Force.fill(0)
    contact.fill(False)
    inside_window.fill(False)
    # assigning v0 to the upper boundary if in shear step
    if t[i]>=T_pressure:vx[jtop:jbottom]=-v0
    # assigning v0 to the upper boundary for uniaxial compression
    if uniaxial_compression:vy[jtop:jbottom]=-v0
    # DEM caluclations
    for jm in range(total_nps-1): #jm: master , js: slave
            for js in range(jm+1,total_nps):
                lx=xc[jm]-xc[js] 
                if abs(lx)<(width-r[jm]-r[js]):
                    distance=np.sqrt(lx**2+(yc[jm]-yc[js])**2) 
                else:
                    distance=np.sqrt((abs(lx)-width)**2+(yc[jm]-yc[js])**2) 

                delta=distance-r[jm]-r[js]
                if delta<0:
                    # finding the normal and tangent unit vectors
                    if abs(lx)<(width-r[jm]-r[js]):
                        normal_x=(xc[jm]-xc[js])/distance
                    else:
                        normal_x=(xc[jm]-xc[js]-np.sign(lx)*width)/distance
                    
                    normal_y=(yc[jm]-yc[js])/distance
                    tangent_x=-normal_y
                    tangent_y=normal_x
                    normal_v=(vx[jm]-vx[js])*normal_x+(vy[jm]-vy[js])*normal_y
                    tangent_v=(vx[jm]-vx[js])*tangent_x+(vy[jm]-vy[js])*tangent_y
                    tangent_v+=-r[jm]*omega[jm]-r[js]*omega[js]
                    theta_rel=(omega[jm]+omega[js])*dt
                    r_avg=(r[jm]+r[js])/2
                    # caluclating forces and updating the accelerations
                    Fn=-Kn*delta-Cn*normal_v 
                    if Fn<0.:
                        Fn=0.
                    Ft=-Kt*tangent_v*dt
                    if abs(Ft)>=mu*Fn:
                        Ft=np.sign(Ft)*mu*Fn
                    Mr=-Kt*r_avg**2*theta_rel
                    if abs(Mr)>=murf*Fn*r_avg:
                        Mr=np.sign(Mr)*murf*Fn*r_avg
                    Fx=Fn*normal_x+Ft*tangent_x 
                    Fy=Fn*normal_y+Ft*tangent_y 
                    # master particle
                    ax[jm]+=Fx/m[jm]
                    ay[jm]+=Fy/m[jm]
                    alpha[jm]+=(-Ft*r[jm]+Mr)/I[jm]
                    # slave particle
                    ax[js]+=-Fx/m[js]
                    ay[js]+=-Fy/m[js]
                    alpha[js]+=(-Ft*r[js]+Mr)/I[js]
                    if boundary[jm]==2:
                        Fy_roller+=Fy
                    if boundary[js]==2:
                        Fy_roller+=-Fy
                    
                    Normal_Force[jm,js]=Fn
                    contact[jm,js]=True

                    # required for periodic boundary stress calc
                    xjm=xc[jm]
                    if xjm<xleft: xjm+=width
                    if xjm>xright: xjm-=width
                    xjs=xc[js]
                    if xjs<xleft: xjs+=width
                    if xjs>xright: xjm-=width
                    
                    # stress calc
                    branch=[xjs-xjm,yc[js]-yc[jm]]
                    if xjm<stress_window_right and xjm>stress_window_left and yc[jm]<stress_window_top and yc[jm]>stress_window_bottom:
                        inside_window[jm]=True
                        
                        if xjs<stress_window_right and xjs>stress_window_left and yc[js]<stress_window_top and yc[js]>stress_window_bottom:
                            inside_window[js]=True
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

                    elif xjs<stress_window_right and xjs>stress_window_left and yc[js]<stress_window_top and yc[js]>stress_window_bottom:
                        inside_window[js]=True
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
    
    # Updating Locations and Velocities
    Fy_roller+=-P*2*sum(r_roller)
    ay_roller=Fy_roller/(m_roller*n_roller)
    for j in range(total_nps):
        if boundary[j]==2:
            if uniaxial_compression: #boundary 2 only moves down based on v0
                yc[j]+=vy[j]*dt
            else: # simple shear with compression (boundary 2 will move left or right based on v0 and can move up and down based on applied pressure)   
                xc[j]+=vx[j]*dt
                ay[j]=ay_roller    
                vy_half[j]+=ay[j]*dt
                yc[j]+=vy_half[j]*dt
                vy[j]=vy_half[j]+ay[j]*dt/2
        elif boundary[j]==1:
            xc[j]+=vx[j]*dt
        else:
            vx_half[j]+=ax[j]*dt
            vy_half[j]+=ay[j]*dt
            omega_half[j]+=alpha[j]*dt
            xc[j]+=vx_half[j]*dt
            yc[j]+=vy_half[j]*dt
            vx[j]=vx_half[j]+ax[j]*dt/2
            vy[j]=vy_half[j]+ay[j]*dt/2
            theta[j]+=omega_half[j]*dt
            omega[j]=omega_half[j]+alpha[j]*dt/2
    
    
    # Plotting
    tplot=0.01
    iplot=int(tplot/dt)
    if i%iplot==0:
        plt.figure(figsize=(10,10))
        nn=32 # number of nodes on each particle
        d_angle=2*math.pi/nn #discretization angle
        plt.clf()
        plt.plot([xleft,xleft],[ytop,ybottom],'k')
        plt.plot([xright,xright],[ytop,ybottom],'k')
        plt.plot([stress_window_left,stress_window_right,stress_window_right,stress_window_left,stress_window_left],[stress_window_bottom,stress_window_bottom,stress_window_top,stress_window_top,stress_window_bottom],'b')
        for j in range(total_nps):
            fcolor='#FFA07A'
            ecolor='#FF8C00'
            if boundary[j]!=0:
                fcolor='#ADD8E6'
                ecolor='#0000A0'
            if inside_window[j]:
                fcolor='#90EE90'
                ecolor='#5A8B2A'
            x_in=[]
            y_in=[]
            x_out=[]
            y_out=[]
            for k in range(nn):
                x=xc[j]+r[j]*math.cos(d_angle*k) 
                y=yc[j]+r[j]*math.sin(d_angle*k)
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
            
            x_dot=xc[j]+r[j]/2*math.cos(theta[j])
            y_dot=yc[j]+r[j]/2*math.sin(theta[j])
            if x_dot<xleft: x_dot+=width
            if x_dot>xright: x_dot-=width
            plt.plot(x_dot,y_dot,'ok',markersize=.5)   
        
        for jm in range(total_nps-1):
            for js in range(jm,total_nps):
                if contact[jm,js]: 
                    xjm=xc[jm]
                    if xjm<xleft: xjm+=width
                    if xjm>xright: xjm-=width
                    xjs=xc[js]
                    if xjs<xleft: xjs+=width
                    if xjs>xright: xjm-=width
                    if abs(xjm-xjs)<(width-r[jm]-r[js]):
                        plt.plot([xjm,xjs],[yc[jm],yc[js]],'k',linewidth=Normal_Force[jm,js]/500)


        plt.axis([xleft-.25*width,xright+.25*width,ybottom-.25*(ytop-ybottom),ytop+.25*(ytop-ybottom)])
        plt.gca().set_aspect('equal')
        plt.savefig('Frame%07d.png' %i)
        plt.close()

# Writing the results to a text file
f=open("config_after_pressure.txt","w+")
#f=open("config_after_shear.txt","w+")
for j in range(total_nps):
    f.write("%f\t%f\t%f\n" % (r[j],xc[j],yc[j]))
f.close()

tanphi=stress_xy/stress_yy
plt.figure(figsize=(10,5))
plt.plot(t,tanphi,'k')
plt.xlabel('time (sec)')
plt.ylabel('tan($phi$)')
plt.axis([0,T,-0.4,0.4])
plt.savefig('stress_ratio.png')
plt.close()

f=open("stress_ratio.txt","w+")
f.write("sigma_xy/sigma_yy\n")
for i in range(len(t)):
    f.write("%f\n" % tanphi[i])
f.close()

# Plotting stresses
plt.figure(figsize=(10,5))
plt.plot(t,-stress_xx,'k',label='$\sigma_{xx}$')
plt.plot(t,-stress_yy,'r',label='$\sigma_{yy}$')
plt.plot(t,-stress_xy,'b',label='$\sigma_{xy}$')
plt.plot(t,-stress_yx,'g',label='$\sigma_{yx}$')
plt.gca().ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.legend()
plt.xlabel('time (sec)')
plt.ylabel('Stress (N/m)')
plt.savefig('stress.png')
plt.close()

print('stress_yy = ',-np.mean(stress_yy),'stress_xy = ',-np.mean(stress_xy))
