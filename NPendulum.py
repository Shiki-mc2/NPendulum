# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2
from numba import jit

matplotlib.interactive(False)

# cal constant
def constF():
    A = np.zeros((n, n))
    B = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            for k in range(max(i,j),n):
                A[i][j] += m[k]
                B[i][j] += m[k]
            if i == j:
                A[i][j] *= l[j]
                B[i][j] *= g
            else:
                A[i][j] *= l[j]
                B[i][j] *= l[j]
    return A,B

@jit
def f(x, v): # Equations of motion
    A = np.copy(A0)
    B = np.copy(B0)
    e = np.full((n,1), -1.0)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                B[i][j] *= np.sin(x[i])
            else:
                A[i][j] *= np.cos(x[i]-x[j])
                B[i][j] *= v[j]**2*np.sin(x[i]-x[j])
    ddX = np.dot( np.linalg.inv(A), np.dot(B, e) )[:,0]

    return ddX

fps = 30.0 # frame per second
dt   = 1/fps*0.005 # dt of cal
dlt  = 1/fps*0.1    # dt of trajectory
dvt  = 1/fps        # second per frame
limT = 30.0
n    = 8 # number of tuple pendulum
g    = 9.8 # Gravitational acceleration
M    = 1.0 # base mass
L    = 1/(n**0.5) # base line length
R    = 20.0 #base marker size
line_wid = 1.0
count_et = 300 # count of emphasis trajectory

graph_bgc   = "#FFFFFF"
title_color = "#000000"

#Initialize physics quantity and drawing param
m    = np.zeros(n) # mass
l    = np.zeros(n) # line length
x    = np.zeros(n) # angle
v    = np.zeros(n) # angular velocity
r    = np.zeros(n) # marker size
c    = np.zeros((n,3)) # circle color

for i in range(n):    
    m[i] = M
    l[i] = L
    x[i] = np.pi*0.45
    v[i] = 0
    r[i] = R
    c[i] = cm.jet(i/(n-1))[:3]

# fpr axis limit
lim  = 0.0
for i in range(n):
    lim  += l[i]
lim +=0.1

plot_x = [0]*n  # position - horizontal  
plot_y = [0]*n  # position - vertical 
plot_x_log = [] # position - horizontal( Trajectory )
plot_y_log = [] # position - vertical  ( Trajectory )

#setting video
fourcc = cv2.VideoWriter_fourcc(*"h264")
video  = cv2.VideoWriter("%s_%2d_%d.mp4" % (sys.argv[0][sys.argv[0].rfind("/")+1:-3], n, time.time()), fourcc, fps, (1600, 900))

# initialize calculation
t    = 0.0
lt   = 0.0
vt   = 0.0
A0, B0 = constF() # cal constant

# initialize figure
plt.rcParams['figure.facecolor'] = graph_bgc
plt.rcParams['axes.facecolor'] = graph_bgc
fig = plt.figure(figsize=(16,9),dpi=100)
plt.subplots_adjust(left=0.04, right=0.96, bottom=0, top=0.955)
plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
plt.tick_params(bottom=False,left=False,right=False,top=False)

sw1 = time.time() #start stop watch
while t < limT:
    t += dt
    
    # Integral (Runge–Kutta method)
    a1 = dt*f(x,v)
    a2 = dt*f(x+dt*v*0.5, v+a1*0.5)
    a3 = dt*f(x+dt*(v+a1*0.5)*0.5, v+a2*0.5)
    a4 = dt*f(x+dt*(v+a2*0.5),     v+a3)
    x = x + dt*(v + (a1 + a2 + a3)/6)
    v = v + (a1 + 2*(a2+a3) + a4)/6

    # angle -> position
    if lt <= t:
        plot_x[0] = l[0]*np.sin(x[0])
        plot_y[0] = -l[0]*np.cos(x[0])
        for i in range(1,n):
            plot_x[i] = plot_x[i-1] + l[i]*np.sin(x[i])
            plot_y[i] = plot_y[i-1] - l[i]*np.cos(x[i])
    
        if len(plot_x_log) == 0:
            plot_x_log = np.array([plot_x])
            plot_y_log = np.array([plot_y])
        else:
            plot_x_log = np.vstack((plot_x_log, plot_x))
            plot_y_log = np.vstack((plot_y_log, plot_y))
        lt +=dlt        

    #plot
    if vt <= t:
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        # set xlim and ylim
        plt.ylim(-lim,lim*1.0/8.0)
        xlim = (lim + lim*1.0/8.0)*8.0/9.0
        plt.xlim(-xlim,xlim)
        
        #Energy
        T = 0 #kinetic energy
        U = 0 #Potential energy
        H = 0 #Hamiltonian
        for i in range(n):
            for j in range(i+1):
                for k in range(i+1):
                    T += 0.5 * m[i] * l[j] * l[k] * v[j] * v[k] * np.cos(x[j]-x[k])
                U -=m[i] * g * l[j] * np.cos(x[j])        
        H = T + U
        
        # Title
        plt.title("$N$=%d $T$=%11.5f $U$=%11.5f $E$=%11.5f" % (n,T,U,H), fontname="monospace",color=title_color,fontsize=20)
        
        # Check Conservation of energy and Process tune
        print ("t=%6.3f H=%12.7e process time=%f" % (t,H,(time.time() - sw1)))
            
        #plot Trajectory
        for i in range(n-1,-1,-1):
            plt.plot(plot_x_log[:-count_et,i],  plot_y_log[:-count_et,i],  color=c[i,:]+(1-c[i,:])*0.7,  linewidth=2.0)
        for i in range(n-1,-1,-1):
            plt.plot(plot_x_log[-count_et:,i],  plot_y_log[-count_et:,i],  color=c[i,:]*0.9,  linewidth=2.0)
        
        #plot Line
        plt.plot([0]+plot_x, [0]+plot_y, color="black", linestyle="-", linewidth=line_wid)
        
        #plot circle(node)
        for i in range(n):
            plt.plot(plot_x[i], plot_y[i], linestyle="none", color=c[i,:], marker="o", markersize=r[i])
        
        #write to video
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer._renderer)
        img = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        video.write(img)
        plt.gca().clear()
        vt +=dvt

print ("T=%13.8e U=%13.8e H=%13.8e" % (T,U,H))
plt.close()
video.release()
