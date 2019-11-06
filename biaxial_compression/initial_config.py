import math
import matplotlib.pyplot as plt

xc=[]
yc=[]
r=0.01
nlayers=21
for i in range(nlayers):
    y=r+3**0.5*r*i
    if i%2==0:
        n=8
        for j in range (n):
            x=r+2*r*j
            xc.append(x)
            yc.append(y)
    else:
        n=7
        for j in range(n):
            x=2*r*(1+j)
            xc.append(x)
            yc.append(y)

# Writing the results to a text file
f=open("initial_config.txt","w+")
for x,y in zip(xc,yc):
    f.write("%f\t%f\t%f\n" % (r,x,y))
f.close()

# Plotting
nn=32
d_angle=2*math.pi/nn

plt.figure(figsize=(10,5))
for x,y in zip(xc,yc):
    xcircle=[]
    ycircle=[]
    for k in range(nn):
        xx=x+math.cos(d_angle*k)*r 
        yy=y+math.sin(d_angle*k)*r 
        xcircle.append(xx)
        ycircle.append(yy)
    plt.fill(xcircle,ycircle,facecolor='#FFA07A',edgecolor='#FF8C00')
    
    #plt.axis([-0.05, 0.35, 0, 0.4])
    #plt.savefig('Frame%07d.png' %i)
plt.gca().set_aspect('equal')
plt.show()


