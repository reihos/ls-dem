# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math
import matplotlib.pyplot as plt

# SI Units

# Particle properties
m=0.04

# Contact properties 
Kn=1e5
Kt=1e5
damping=0.05
Ccrit=2*np.sqrt(Kn*m)
Cn=damping*Ccrit
mu=0.5
murf=0.1
g=9.81

# Geometrical properties
ybottom=0.
ytop=0.68
xright=1.
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
T=1.2
t=np.arange(0,T,dt)
T_pressure=0.

# location
#f=open("initial_config_999.txt","r")
f=open("initial_config_999.txt","r")
lines=f.read().splitlines()
total_nps=len(lines)
r=np.zeros(total_nps)
xc=np.zeros((len(t),total_nps)) # x of center
yc=np.zeros((len(t),total_nps)) # y of center
j=0
for line in lines:
    s=line.split()
    r[j]=s[0]
    xc[0,j]=s[1]
    yc[0,j]=s[2]
    j+=1
print('number of particles =',total_nps)

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


# Updating the Kinematic Properties of Particles
for i in range(len(t)-1):
    print(i)
    for jm in range(total_nps-1): #jm: master , js: slave
        for js in range(jm+1,total_nps):
            lx=xc[i,jm]-xc[i,js] 
            distance=np.sqrt((xc[i,jm]-xc[i,js])**2+(yc[i,jm]-yc[i,js])**2) 
            delta=distance-r[jm]-r[js]
            if delta<0:
                # finding the normal and tangent unit vectors
                normal_x=(xc[i,jm]-xc[i,js])/distance
                normal_y=(yc[i,jm]-yc[i,js])/distance
                tangent_x=-normal_y
                tangent_y=normal_x
                normal_v=(vx[i,jm]-vx[i,js])*normal_x+(vy[i,jm]-vy[i,js])*normal_y
                tangent_v=(vx[i,jm]-vx[i,js])*tangent_x+(vy[i,jm]-vy[i,js])*tangent_y
                tangent_v+=-r[jm]*omega[i,jm]-r[js]*omega[i,js]
                theta_rel=(omega[i,jm]+omega[i,js])*dt
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
                ax[i,jm]+=Fx/m
                ay[i,jm]+=Fy/m
                alpha[i,jm]+=(-Ft*r[jm]+Mr)/I[jm]
                # slave particle
                ax[i,js]+=-Fx/m
                ay[i,js]+=-Fy/m
                alpha[i,js]+=(-Ft*r[js]+Mr)/I[js]
                    
               #     branch=[xc[i,js]-xc[i,jm],yc[i,js]-yc[i,jm]]
               #     if xc[i,jm]<stress_window_right and xc[i,jm]>stress_window_left and yc[i,jm]<stress_window_top and yc[i,jm]>stress_window_bottom:
               #         inside_window[i,jm]=True
               #         
               #         if xc[i,js]<stress_window_right and xc[i,js]>stress_window_left and yc[i,js]<stress_window_top and yc[i,js]>stress_window_bottom:
               #             inside_window[i,js]=True
               #             stress_xx[i]+=Fx*branch[0]
               #             stress_yy[i]+=Fy*branch[1]
               #             stress_xy[i]+=Fy*branch[0]
               #             stress_yx[i]+=Fx*branch[1]
               #         else:
               #             scale=distance*r[jm]
               #             stress_xx[i]+=Fx*branch[0]*scale
               #             stress_yy[i]+=Fy*branch[1]*scale
               #             stress_xy[i]+=Fy*branch[0]*scale
               #             stress_yx[i]+=Fx*branch[1]*scale

               #     elif xc[i,js]<stress_window_right and xc[i,js]>stress_window_left and yc[i,js]<stress_window_top and yc[i,js]>stress_window_bottom:
               #         inside_window[i,js]=True
               #         scale=distance*r[js]
               #         stress_xx[i]+=Fx*branch[0]*scale
               #         stress_yy[i]+=Fy*branch[1]*scale
               #         stress_xy[i]+=Fy*branch[0]*scale
               #         stress_yx[i]+=Fx*branch[1]*scale
        
        delta_wall=yc[i,jm]-ybottom-r[jm]
        if delta_wall<0:
            normal_x=0.
            normal_y=1.
            tangent_x=-normal_y
            tangent_y=normal_x
            normal_v=(vx[i,jm])*normal_x+(vy[i,jm])*normal_y
            tangent_v=(vx[i,jm])*tangent_x+(vy[i,jm])*tangent_y
            tangent_v+=-r[jm]*omega[i,jm]
            theta_rel=(omega[i,jm])*dt
            r_avg=r[jm]
            # caluclating forces and updating the accelerations
            Fn=-Kn*delta_wall-Cn*normal_v 
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
            ax[i,jm]+=Fx/m
            ay[i,jm]+=Fy/m
            alpha[i,jm]+=(-Ft*r[jm]+Mr)/I[jm]

                            

    # calculating the stresses in the stress window
    #stress_xx[i]/=window_area
    #stress_yy[i]/=window_area
    #stress_xy[i]/=window_area
    #stress_yx[i]/=window_area
    
    # Updating Locations and Velocities
    for j in range(total_nps):
        vx_half[j]+=ax[i,j]*dt
        vy_half[j]+=ay[i,j]*dt
        omega_half[j]+=alpha[i,j]*dt
        xc[i+1,j]=xc[i,j]+vx_half[j]*dt
        yc[i+1,j]=yc[i,j]+vy_half[j]*dt
        vx[i+1,j]=vx_half[j]+ax[i,j]*dt/2
        vy[i+1,j]=vy_half[j]+ay[i,j]*dt/2
        theta[i+1,j]=theta[i,j]+omega_half[j]*dt
        omega[i+1,j]=omega_half[j]+alpha[i,j]*dt/2

# Writing the results to a text file
#f=open("config_after_pressure.txt","w+")
#for j in range(total_nps):
#    f.write("%f\t%f\t%f\n" % (r[j],xc[-1,j],yc[-1,j]))
#f.close()

#print('stress_xx = ',-np.mean(stress_xx),'stress_xy = ',-np.mean(stress_xy))

# Plotting
#plt.figure(1,figsize=(10,5))
#plt.plot(t[:-1],-stress_xx,'k',label='$\sigma_{xx}$')
#plt.plot(t[:-1],-stress_yy,'r',label='$\sigma_{yy}$')
#plt.plot(t[:-1],-stress_xy,'b',label='$\sigma_{xy}$')
#plt.plot(t[:-1],-stress_yx,'g',label='$\sigma_{yx}$')
#plt.gca().ticklabel_format(axis='both',style='sci',scilimits=(0,0))
#plt.legend()
#plt.xlabel('time (sec)')
#plt.ylabel('Stress (N/m)')
#plt.savefig('stress.png')

plt.figure(figsize=(15,5))
# plotting every 0.01 sec
nn=32 # number of nodes on each particle
d_angle=2*math.pi/nn #discretization angle
for i in range(0,int(T/dt),int(0.01/dt)): 
    plt.clf()
    #plt.plot([stress_window_left,stress_window_right,stress_window_right,stress_window_left,stress_window_left],[stress_window_bottom,stress_window_bottom,stress_window_top,stress_window_top,stress_window_bottom],'b')
    for j in range(total_nps):
        fcolor='#FFA07A'
        ecolor='#FF8C00'
        #if inside_window[i,j]:
        #    fcolor='#90EE90'
        #    ecolor='#5A8B2A'
        x_in=[]
        y_in=[]
        for k in range(nn):
            x=xc[i,j]+r[j]*math.cos(d_angle*k) 
            y=yc[i,j]+r[j]*math.sin(d_angle*k)
            x_in.append(x)
            y_in.append(y)
        plt.fill(x_in,y_in,facecolor=fcolor,edgecolor=ecolor)
        
        x_dot=xc[i,j]+r[j]/2*math.cos(theta[i,j])
        y_dot=yc[i,j]+r[j]/2*math.sin(theta[i,j])
        plt.plot(x_dot,y_dot,'ok',markersize=.5)   
    
    plt.axis([xleft-2*width,xright+2*width,ybottom-.25*(ytop-ybottom),ytop+.25*(ytop-ybottom)])
    plt.gca().set_aspect('equal')
    plt.savefig('Frame%07d.png' %i)
