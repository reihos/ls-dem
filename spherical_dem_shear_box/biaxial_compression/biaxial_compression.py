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
y_roller_top[:]=0.53
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
T=0.1
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
boundary=np.zeros(total_nps,int) # top and bottom
side_boundary=np.zeros(total_nps,bool) # sides
v0=1
P=10000
for j in range(total_nps):
    if j>=jbottom: #bottom
        boundary[j]=1
    elif j>=jtop: #top
        boundary[j]=2
        vy[j]=-v0

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
    
    # DEM caluclations
    for jm in range(total_nps-1): #jm: master , js: slave
            for js in range(jm+1,total_nps):
                distance=np.sqrt((xc[jm]-xc[js])**2+(yc[jm]-yc[js])**2) 
                delta=distance-r[jm]-r[js]
                if delta<0:
                    # finding the normal and tangent unit vectors
                    normal_x=(xc[jm]-xc[js])/distance
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
                    
                    # force chains
                    Normal_Force[jm,js]=Fn
                    contact[jm,js]=contact[js,jm]=True
                
                    # stress calc
                    branch=[xc[js]-xc[jm],yc[js]-yc[jm]]
                    if xc[jm]<stress_window_right and xc[jm]>stress_window_left and yc[jm]<stress_window_top and yc[jm]>stress_window_bottom:
                        inside_window[jm]=True
                        
                        if xc[js]<stress_window_right and xc[js]>stress_window_left and yc[js]<stress_window_top and yc[js]>stress_window_bottom:
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

                    elif xc[js]<stress_window_right and xc[js]>stress_window_left and yc[js]<stress_window_top and yc[js]>stress_window_bottom:
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
    
    if i==0:
        side_boundary_left_list=[]
        side_boundary_right_list=[]
        # Pressure boundary caluclations
        side_boundary=np.zeros(total_nps,bool) # sides
        # left
        # finding the bottommost boundary particles
        xmin=xright
        for j in range(total_nps):
            if boundary[j]==0 and yc[j]-r[j]<ybottom:
                if xc[j]<xmin:
                    jleft=j
                    xmin=xc[j]
        side_boundary_left=[jleft,xc[jleft],yc[jleft]-r[jleft]]
        side_boundary[jleft]=True
        side_boundary_left_list.append(side_boundary_left)

        # finding the side particles
        for n in range(total_nps):
            for j in range(total_nps):
                jb=0
                if j==side_boundary_left[0]:
                    xmin=xright
                    for jj in range(total_nps):
                        if contact[j,jj] and boundary[jj]==0 and side_boundary[jj]==False:
                            if xc[jj]-r[jj]<xmin:
                                xmin=xc[jj]-r[jj]
                                jb=jj
                    break
            
            jleft=side_boundary_left[0] 
            distance=np.sqrt((xc[jb]-xc[jleft])**2+(yc[jb]-yc[jleft])**2)
            scale=distance*r[jleft]
            branch=[xc[jb]-xc[jleft],yc[jb]-yc[jleft]]
            xcontact=xc[jleft]+branch[0]*scale
            ycontact=yc[jleft]+branch[1]*scale
            lx=xcontact-side_boundary_left[1]
            ly=ycontact-side_boundary_left[2]
            ax[jleft]+=P*ly/m[jm]
            ay[jleft]+=-P*lx/m[jm]
            if yc[jb]+r[jb]<yc[jleft]+r[jleft]:
                break
            side_boundary_left=[jb,xcontact,ycontact]
            side_boundary[jb]=True
            side_boundary_left_list.append(side_boundary_left)

        # right 
        # finding the bottommost boundary particles
        xmax=xleft
        for j in range(total_nps):
            if boundary[j]==0 and yc[j]-r[j]<ybottom:
                if xc[j]>xmax:
                    jright=j
                    xmax=xc[j]
        side_boundary_right=[jright,xc[jright],yc[jright]-r[jright]]
        side_boundary[jright]=True
        side_boundary_right_list.append(side_boundary_right)
        
        # finding the side particles
        for n in range(total_nps):
            backward=False
            for j in range(total_nps):
                jb=0
                if j==side_boundary_right[0]:
                    xmax=xleft
                    for jj in range(total_nps):
                        if contact[j,jj] and boundary[jj]==0 and side_boundary[jj]==False:
                            if xc[jj]+r[jj]>xmax:
                                xmax=xc[jj]+r[jj]
                                jb=jj
                    
                    # the case were the boundary particle is not in contact with any particle other than the previuos boundary particle
                    #if jb==-1:
                    #    backward=True
                    #    for jj in range(total_nps):
                    #        if contact[j,jj] and boundary[jj]==0:
                    #            if xc[jj]+r[jj]>xmax:
                    #                xmax=xc[jj]+r[jj]
                    #                jb=jj
                    #print(jb)
                    break
            
            jright=side_boundary_right[0] 
            distance=np.sqrt((xc[jb]-xc[jright])**2+(yc[jb]-yc[jright])**2)
            scale=distance*r[jright]
            branch=[xc[jb]-xc[jright],yc[jb]-yc[jright]]
            xcontact=xc[jright]+branch[0]*scale
            ycontact=yc[jright]+branch[1]*scale
            lx=xcontact-side_boundary_right[1]
            ly=ycontact-side_boundary_right[2]
            ax[jright]+=-P*ly/m[jm]
            ay[jright]+=P*lx/m[jm]
            if yc[jb]+r[jb]<yc[jright]+r[jright]:
                break
            side_boundary_right=[jb,xcontact,ycontact]
            side_boundary[jb]=True
            side_boundary_right_list.append(side_boundary_right)
    
    else:
        #left
        for n in range(len(side_boundary_left_list)):
            jleft=side_boundary_left_list[n][0]

            #for the last particle
            if n==len(side_boundary_left_list)-1:
                lx=xc[jleft]-side_boundary_left_list[n][1]
                ly=yc[jleft]+r[jleft]-side_boundary_left_list[n][2]

            else:
                lx=side_boundary_left_list[n+1][1]-side_boundary_left_list[n][1]
                ly=side_boundary_left_list[n+1][2]-side_boundary_left_list[n][2]
            ax[jleft]+=P*ly/m[jleft]
            ay[jleft]+=-P*lx/m[jleft]
        #right
        for n in range(len(side_boundary_right_list)):
            jright=side_boundary_right_list[n][0]

            #for the last particle
            if n==len(side_boundary_right_list)-1:
                lx=xc[jright]-side_boundary_right_list[n][1]
                ly=yc[jright]+r[jright]-side_boundary_right_list[n][2]
            else:
                lx=side_boundary_right_list[n+1][1]-side_boundary_right_list[n][1]
                ly=side_boundary_right_list[n+1][2]-side_boundary_right_list[n][2]
            ax[jright]+=-P*ly/m[jright]
            ay[jright]+=P*lx/m[jright]
        
        # Updating Locations and Velocities
    for j in range(total_nps):
        if boundary[j]==2:
            yc[j]+=vy[j]*dt
        elif boundary[j]!=1:
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
    tplot=0.001
    iplot=int(tplot/dt)
    if i%iplot==0:
        plt.figure(figsize=(10,10))
        nn=32 # number of nodes on each particle
        d_angle=2*math.pi/nn #discretization angle
        plt.clf()
        #plt.plot([xleft,xleft],[ytop,ybottom],'k')
        #plt.plot([xright,xright],[ytop,ybottom],'k')
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
            if side_boundary[j]:    
                fcolor='#00316E'
                ecolor='#009DFF'
            
            x_in=[]
            y_in=[]
            for k in range(nn):
                x=xc[j]+r[j]*math.cos(d_angle*k) 
                y=yc[j]+r[j]*math.sin(d_angle*k)
                x_in.append(x)
                y_in.append(y)
            
            plt.fill(x_in,y_in,facecolor=fcolor,edgecolor=ecolor)
            
            x_dot=xc[j]+r[j]/2*math.cos(theta[j])
            y_dot=yc[j]+r[j]/2*math.sin(theta[j])
            plt.plot(x_dot,y_dot,'ok',markersize=.5)   
        
        # force chains
        for jm in range(total_nps-1):
            for js in range(jm,total_nps):
                if contact[jm,js]: 
                    plt.plot([xc[jm],xc[js]],[yc[jm],yc[js]],'k',linewidth=Normal_Force[jm,js]/100)


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

print('stress_yy = ',-np.mean(stress_yy),'stress_xx = ',-np.mean(stress_xx),'stress_xy = ',-np.mean(stress_xy))
