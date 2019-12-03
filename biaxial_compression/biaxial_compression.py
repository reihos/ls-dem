# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math 
import matplotlib.pyplot as plt

# SI Units

# Particle properties
total_nps=158 #total number of particles
rho=7.85*1e3 #kg/m^2
#r=np.zeros(total_nps)
r=np.ones(total_nps)*0.003
r_avg=np.mean(r)
m=math.pi*r[0]**2*rho
I=0.5*m*r[0]**2

# Contact properties 
Kn=Kt=5*1e7 #N/m
damping=0.05
Ccrit=2*math.sqrt(Kn*m)
Cn=damping*Ccrit
mu=0.5

# Geometrical properties
ybottom=0.
ytop=0.110
xright=0.048
xleft=0.

# Window for Stress Calculation
stress_window_top=0.75*(ytop-ybottom)+ybottom
stress_window_bottom=0.25*(ytop-ybottom)+ybottom
stress_window_right=0.75*(xright-xleft)+xleft
stress_window_left=0.25*(xright-xleft)+xleft
window_area=(stress_window_top-stress_window_bottom)*(stress_window_right-stress_window_left)

# Loading condition
P = 50662.5

# Discretization
dt=.000001
dtcrit=2*(math.sqrt(1+damping**2)-damping)/math.sqrt(Kn/m)
if dt>0.1*dtcrit:
    print("critical timestep exceeded")
print('the ciritical timestep is {0:f} and the timestep used is {1:f}'.format(dtcrit,dt))
dx=dy=0.001
nn=32 # number of nodes on each particle
d_angle=2*math.pi/nn #discretization angle

# Initializing t, x, y, v, and omega arrays
# time
T=.002
t=np.arange(0,T,dt)

# location
xc=np.zeros((len(t),total_nps)) # x of center
yc=np.zeros((len(t),total_nps)) # y of center

f=open("initial_config.txt","r")
for j in range(total_nps):
    a=f.readline()
    b=a.split()
    #r[j]=b[0]
    xc[0,j]=b[1]
    yc[0,j]=b[2]

x=np.zeros((len(t),total_nps,nn)) 
y=np.zeros((len(t),total_nps,nn))
#r_avg=np.mean(r)

for j in range(total_nps):
    for k in range(nn):
        x[0,j,k]=xc[0,j]+r[j]*math.cos(d_angle*k)
        y[0,j,k]=yc[0,j]+r[j]*math.sin(d_angle*k)

# velocity
vx=np.zeros((len(t),total_nps)) #velocity of the center of mass
vy=np.zeros((len(t),total_nps)) 
vx_half=np.zeros(total_nps)
vy_half=np.zeros(total_nps)
omega=np.zeros((len(t),total_nps))
omega_half=np.zeros(total_nps)

# acceleration
ax=np.zeros((len(t),total_nps))
ay=np.zeros((len(t),total_nps))
alpha=np.zeros((len(t),total_nps))


# Setting the Boundary Particles
boundary=np.zeros(total_nps)
v0=5.
for j in range(total_nps):
    if yc[0,j]<2*r_avg:
        vy[:,j]=0.
        boundary[j]=1
    if yc[0,j]>ytop-2*r_avg:
        vy[:,j]=-v0
        boundary[j]=2

# Defining the Level Set Functions
ny=int(round((ytop-ybottom)/dy))
nx=int(round((xright-xleft)/dx))
xmid=(xleft+xright)/2
ymid=(ybottom+ytop)/2
ixmid=int(nx/2)
iymid=int(ny/2)
LS=np.zeros((nx+1,ny+1,total_nps))
for xx in range(nx+1):
    for yy in range(ny+1):
        xg=xleft+xx*dx # x of grid point
        yg=ybottom+yy*dy
        for j in range(total_nps):
            R=math.sqrt((xg-xmid)**2+(yg-ymid)**2) # distance from grid point to center of the particle
            LS[xx,yy,j]=R-r[0]
LSi=0.

# Initializing stress arrays
stress_xx=np.zeros(len(t)-1)
stress_yy=np.zeros(len(t)-1)
stress_xy=np.zeros(len(t)-1)
stress_yx=np.zeros(len(t)-1)
inside_window=np.zeros((len(t)-1,total_nps),bool)


# Updating the Kinematic Properties of Particles
for i in range(len(t)-1):
    print(i)
    # finding the bottommost left and right boundary particles
    xmin=xright
    xmax=xleft
    for j in range(total_nps):
        if yc[0,j]<0.02:
            if xc[0,j]<xmin:
                ileft_bound=j
                xmin=xc[0,j]
            if xc[0,j]>xmax:
                iright_bound=j
                xmax=xc[0,j]
    surrounding_bound_particle_left=[[ileft_bound,[xc[0,ileft_bound],yc[0,ileft_bound]-r[ileft_bound]]]]
    surrounding_bound_particle_right=[[iright_bound,[xc[0,iright_bound],yc[0,iright_bound]-r[iright_bound]]]]
    
    # looping across the master particles
    for jm in range(total_nps-1): #jm: master , js: slave
        num = 0
        # looping across the slave particle
        for js in range(jm+1,total_nps):
            new_particle_left = True
            new_particle_right = True
            Fn_max=0.
            Fx_max=0.
            Fy_max=0.
            # looping across nodes of the master particle
            for k in range(nn):
                # updating LS indices
                # relative indices
                ixf=int(math.floor((x[i,jm,k]-xc[i,js])/dx))
                iyf=int(math.floor((y[i,jm,k]-yc[i,js])/dy))
                xf=xc[i,js]+ixf*dx
                yf=yc[i,js]+iyf*dy
                # absolute indices to be used for LS
                ixf+=ixmid
                iyf+=iymid
                if ixf<0 or ixf>=nx:
                    continue
                # interpolating
                LSi=LS[ixf,iyf,js]*(xf+dx-x[i,jm,k])*(yf+dy-y[i,jm,k])+LS[ixf+1,iyf,js]*(x[i,jm,k]-xf)*(yf+dy-y[i,jm,k])+LS[ixf,iyf+1,js]*(xf+dx-x[i,jm,k])*(y[i,jm,k]-yf)+LS[ixf+1,iyf+1,js]*(x[i,jm,k]-xf)*(y[i,jm,k]-yf)
                LSi=LSi/dx/dy
                
                if LSi<0:
                    # finding the normal direction
                    dLS_dx=((yf+dy-y[i,jm,k])*(LS[ixf+1,iyf,js]-LS[ixf,iyf,js])+(y[i,jm,k]-yf)*(LS[ixf+1,iyf+1,js]-LS[ixf,iyf+1,js]))/dx/dy
                    dLS_dy=((xf+dx-x[i,jm,k])*(LS[ixf,iyf+1,js]-LS[ixf,iyf,js])+(x[i,jm,k]-xf)*(LS[ixf+1,iyf+1,js]-LS[ixf+1,iyf,js]))/dx/dy
                    mag_dLS=math.sqrt(dLS_dx**2+dLS_dy**2)
                    normal_x=dLS_dx/mag_dLS
                    normal_y=dLS_dy/mag_dLS
                    tangent_x=-normal_y
                    tangent_y=normal_x
                    normal_v=(vx[i,jm]-vx[i,js])*normal_x+(vy[i,jm]-vy[i,js])*normal_y #the direction of normal is outward from the slave particle
                    tangent_v=(vx[i,jm]-vx[i,js])*tangent_x+(vy[i,jm]-vy[i,js])*tangent_y #the direction of tangent is in a way that n x t is upwards (if n is going left t is going down)
                    tangent_v+=-r[jm]*omega[i,jm]-r[js]*omega[i,js]
                    
                    # updating the accelerations
                    Fn=-Kn*LSi-Cn*normal_v 
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
                    alpha[i,jm]+=-Ft/I
                    # slave particle
                    ax[i,js]+=-Fx/m
                    ay[i,js]+=-Fy/m
                    alpha[i,js]+=-Ft/I

                    if x[i,jm,k]<stress_window_right and x[i,jm,k]>stress_window_left and y[i,jm,k]<stress_window_top and y[i,jm,k]>stress_window_bottom:
                        inside_window[i,jm]=True
                        inside_window[i,js]=True
                        if Fn>Fn_max:
                            Fn_max=Fn
                            Fx_max=Fx 
                            Fy_max=Fy  

                # checking for the surrounding particles of the bounadry parictle:
                # left
                if LSi<1e-3 and jm==ileft_bound and boundary[jm]!=2:
                    if new_particle_left:
                        surrounding_bound_particle_left.append([js,[[x[i,jm,k],y[i,jm,k]]]])
                        new_particle_left = False
                        num+=1
                    else:
                        surrounding_bound_particle_left[num][1].append([x[i,jm,k],y[i,jm,k]])
                
                # right
                if LSi<1e-3 and jm==iright_bound and boundary[jm]!=2:
                    # checking for right bounadry paricles
                    if new_particle_right:
                        surrounding_bound_particle_right.append([js,[[x[i,jm,k],y[i,jm,k]]]])
                        new_particle_right = False
                        num+=1
                    else:
                        surrounding_bound_particle_right[num][1].append([x[i,jm,k],y[i,jm,k]])
            
            # caluclating the stress between particles jm and js in the stress window
            branch=[xc[i,js]-xc[i,jm],yc[i,js]-yc[i,jm]]
            stress_xx[i]+=Fx_max*branch[0]
            stress_yy[i]+=Fy_max*branch[1]
            stress_xy[i]+=Fy_max*branch[0]
            stress_yx[i]+=Fx_max*branch[1]
    
        
        #print(jm,ax[i,jm],ay[i,jm])
        #print(i,jm,surrounding_bound_particle_left,surrounding_bound_particle_right)

        # applying pressure to the boundary particle
        # left
        if jm==ileft_bound and boundary[jm]!=2:
            xmin = xright
            xsum = 0.
            ysum = 0.
            counter = 0
            for bound in surrounding_bound_particle_left[1:]:
                if xc[i,bound[0]]< xmin:
                    xmin=xc[i,bound[0]]
                    ileft_bound=bound[0]
            for bound in surrounding_bound_particle_left:
                if bound[0]==ileft_bound:
                    for coords in bound[1]:
                        xsum += coords[0]
                        ysum += coords[1]
                        counter +=1
                    xavg=xsum/counter
                    yavg=ysum/counter
            lx=xavg-surrounding_bound_particle_left[0][1][0]
            ly=yavg-surrounding_bound_particle_left[0][1][1]
            ax[i,jm]+=ly*P/m
            ay[i,jm]+=-lx*P/m
            #print(jm,lx,ly,ax[i,jm],ay[i,jm])
            surrounding_bound_particle_left=[[ileft_bound,[xavg,yavg]]]
            #print('left',surrounding_bound_particle_left)

        # right
        if jm==iright_bound and boundary[jm]!=2:
            xmax = xleft
            xsum = 0.
            ysum = 0.
            counter = 0
            for bound in surrounding_bound_particle_right[1:]:
                if xc[i,bound[0]]> xmax:
                    xmax=xc[i,bound[0]]
                    iright_bound=bound[0]
            for bound in surrounding_bound_particle_right:
                if bound[0]==iright_bound:
                    for coords in bound[1]:
                        xsum += coords[0]
                        ysum += coords[1]
                        counter +=1
                    xavg=xsum/counter
                    yavg=ysum/counter
            lx=xavg-surrounding_bound_particle_right[0][1][0]
            ly=yavg-surrounding_bound_particle_right[0][1][1]
            ax[i,jm]+=-ly*P/m
            ay[i,jm]+=lx*P/m
            surrounding_bound_particle_right=[[iright_bound,[xavg,yavg]]]
            #print('right',surrounding_bound_particle_right)
            
    # calculating the stresses in the stress window
    stress_xx[i]/=window_area
    stress_yy[i]/=window_area
    stress_xy[i]/=window_area
    stress_yx[i]/=window_area
    
    # updating locations and velocities
    for j in range(total_nps):
        #print(j,boundary[j])
        if boundary[j]!=0:
            xc[i+1,j]=xc[i,j]
            yc[i+1,j]=yc[i,j]+vy[i,j]*dt
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

plt.figure(2,figsize=(10,5))
for i in range(0,int(T/dt),int(0.0001/dt)): 
    plt.clf()
    plt.plot([xleft,xleft,xright,xright],[ytop,ybottom,ybottom,ytop],'k')
    plt.plot([stress_window_left,stress_window_right,stress_window_right,stress_window_left,stress_window_left],[stress_window_bottom,stress_window_bottom,stress_window_top,stress_window_top,stress_window_bottom],'b')
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
            if abs(x[i,j,k]-x[i,j,0])>0.05:
                x_out.append(x[i,j,k])
                y_out.append(y[i,j,k])
            else:
                x_in.append(x[i,j,k])
                y_in.append(y[i,j,k])
            plt.fill(x_out,y_out,facecolor=fcolor,edgecolor=ecolor)
            plt.fill(x_in,y_in,facecolor=fcolor,edgecolor=ecolor)
        
    
    plt.axis([-0.02,0.07, 0, 0.12])
    plt.gca().set_aspect('equal')
    plt.savefig('Frame%07d.png' %i)

