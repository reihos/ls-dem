# Updating the location of a bouncing football using molecular dynamic concepts

import numpy as np
import math
import matplotlib.pyplot as plt

# SI Units
g=9.81
# Particle properties
rho=2650

# Contact properties 
Kn=1.6e6
Kt=1e6
COR=0.6 # coeff of restitution
damping=-math.log(COR)/math.sqrt(math.pi**2+math.log(COR)**2)
#damping=0.05
mu=0.5317
murf=0.01


# Geometrical properties
ybottom=0.
ytop=40/1000
xleft=0.
xright=50/1000
width=xright-xleft

# Discretization
dt=1e-5

# Initializing t, x, y, v, omega, and energy arrays
# time
T=0.5
t=np.arange(0,T,dt)
f=open("a08d83.data","r")
lines=f.read().splitlines()
f.close()
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
r/=1000 # converting from mm to m
xc/=1000
yc/=1000
print('number of particles =',total_nps)
m=rho*np.pi*r**2
I=0.5*m*r**2
Ccrit=2*np.sqrt(Kn*np.average(m))
Cn=damping*Ccrit
dtcrit=2*(math.sqrt(1+damping**2)-damping)/math.sqrt(Kn/np.average(m))
#if dt>0.1*dtcrit:
#    raise RuntimeError("critical timestep exceeded")
print('the ciritical timestep is {0:f} and the timestep used is {1:f}'.format(dtcrit,dt))

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

Ek=np.zeros(len(t))
run_out=np.zeros(len(t))
run_out[0]=max(xc[0])

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
                ax[i,jm]+=Fx/m[jm]
                ay[i,jm]+=Fy/m[jm]
                alpha[i,jm]+=(-Ft*r[jm]+Mr)/I[jm]
                # slave particle
                ax[i,js]+=-Fx/m[js]
                ay[i,js]+=-Fy/m[js]
                alpha[i,js]+=(-Ft*r[js]+Mr)/I[js]
                    
        delta_wall=abs(yc[i,jm]-ybottom)-r[jm] # bottom boundary
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
            ax[i,jm]+=Fx/m[jm]
            ay[i,jm]+=Fy/m[jm]
            alpha[i,jm]+=(-Ft*r[jm]+Mr)/I[jm]

        delta_wall=abs(xc[i,jm]-xleft)-r[jm] # left boundary
        if delta_wall<0:
            normal_x=1.
            normal_y=0.
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
            ax[i,jm]+=Fx/m[jm]
            ay[i,jm]+=Fy/m[jm]
            alpha[i,jm]+=(-Ft*r[jm]+Mr)/I[jm]
                            

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

        # Caluclating Kinetic Energy and run_out
        Ek[i+1]=0.5*m[j]*(vx[i,j]**2+vy[i,j]**2)
    run_out[i+1]=max(xc[i,:])

# Writing the results to a text file
f=open("result.txt","w+")
for j in range(total_nps):
    f.write("%f\t%f\t%f\n" % (r[j],xc[-1,j],yc[-1,j]))
f.close()

#Plotting
plt.figure()
plt.plot(t,run_out)
plt.savefig("run_out.png")
plt.figure()
plt.plot(t,Ek)
plt.savefig("kinetic energy.png")

plt.figure(figsize=(15,5))
nn=32 # number of nodes on each particle
d_angle=2*math.pi/nn #discretization angle
for i in range(0,int(T/dt),int(0.01/dt)): 
    plt.clf()
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
    
    plt.axis([xleft,xright+2.5*width,ybottom,ytop+.25*(ytop-ybottom)])
    plt.gca().set_aspect('equal')
    plt.savefig('Frame%07d.png' %i)
