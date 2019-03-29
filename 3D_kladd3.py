# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:47:59 2019

@author: hans.halvorsen
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:53:29 2019

@author: hans.halvorsen
"""

import numpy as np                
import matplotlib.pyplot as plt 
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation
    
def vecToMat(x,M):
    xCo = x % M
    yCo = int(x / M)
    return xCo, yCo  
    
def matToVec(xCo,yCo,M):
    return xCo + M*yCo

def backwards_euler(t_step,w_k,b,big_id,A_w,uvsq):
    A = big_id-t_step*A_w
    x = w_k + t_step*b + t_step*uvsq(w_k)
    w_next = spsolve(A,x)
    return w_next

def forward_euler(r_u,r_v,u_i,v_i,A,f,k,h):    
    uvv = u_i*v_i*v_i
    
    u = u_i + (r_u/h**2)*A.dot(u_i) - uvv +f*(1-u_i)
    v = v_i + (r_v/h**2)*A.dot(v_i) + uvv -(f+k)*v_i
    return u, v

def f_u(M):
    #f = np.sin(x)
    f = np.ones((M+1,M+1)) - 0.02*np.random.random((M+1,M+1))
    sq_rad = int((M+1)/8)
    M_half = int((M+1)/2)
    f[(M_half - sq_rad):(M_half + sq_rad),(M_half - sq_rad):(M_half + sq_rad)] = .5 
    f = np.reshape(f,(M+1)**2)
    return f

def f_v(M):
    #f = np.cos(x)
    f = 0.02*np.random.random((M+1,M+1))
    sq_rad = int((M+1)/8)
    M_half = int((M+1)/2)
    f[(M_half - sq_rad):(M_half + sq_rad),(M_half - sq_rad):(M_half + sq_rad)] = .25 
    f = np.reshape(f,(M+1)**2)
    return np.asarray(f)

def Dirichlet_mat(M):
    print('Building A...')
    e=np.ones((M+1)**2)
    e2=([1]*(M)+[0])*(M+1)
    e3=([0]+[1]*(M))*(M+1)
    A=sp.spdiags([-4*e,e2,e3,e,e],[0,-1,1,-(M+1),(M+1)],(M+1)**2,(M+1)**2)
    print('Finished building A!')
    return A
    
def Neumann_mat(M):
    diag = [-4]*(M+1)**2
    matrix = sp.spdiags([diag],[0],(M+1)**2,(M+1)**2)
    matrix = sp.lil_matrix(matrix)
    for s in range((M+1)**2):
        if s % 5000 == 0:
            print('developing matrix...' , s)
        i,j = vecToMat(s,M+1)
        if i == 0:
            if j == 0:
                matrix[s,s+1] = 2
                matrix[s,matToVec(i,j+1,M+1)] = 2
            elif j == M:
                matrix[s,s+1] = 2
                matrix[s,matToVec(i,j-1,M+1)] = 2
            else:
                matrix[s,s+1] = 2
                matrix[s,matToVec(i,j+1,M+1)] = 1
                matrix[s,matToVec(i,j-1,M+1)] = 1
        elif i == M:
            if j == 0:
                matrix[s,s-1] = 2
                matrix[s,matToVec(i,j+1,M+1)] = 2
            elif j == M:
                matrix[s,s-1] = 2
                matrix[s,matToVec(i,j-1,M+1)] = 2
            else:
                matrix[s,s-1] = 2
                matrix[s,matToVec(i,j+1,M+1)] = 1
                matrix[s,matToVec(i,j-1,M+1)] = 1
        elif j == 0 and i != 0 and i != M:
            matrix[s,s+1] = 1
            matrix[s,s-1] = 1
            matrix[s,matToVec(i,j+1,M+1)] = 2
        elif j == M and i != 0 and i != M:
            matrix[s,s+1] = 1
            matrix[s,s-1] = 1
            matrix[s,matToVec(i,j-1,M+1)] = 2
        else:
            matrix[s,s+1] = 1
            matrix[s,s-1] = 1
            matrix[s,matToVec(i,j+1,M+1)] = 1
            matrix[s,matToVec(i,j-1,M+1)] = 1
    
    
    return sp.csr_matrix(matrix)

def main():
    M = 200
    h = 1/M
    T = 5000
    x = np.linspace(0,1,M+1)
    y = np.linspace(0,1,M+1)
    
    r_u = 0.12*h**2
    r_v = 0.08*h**2
    f   = 0.020
    k   = 0.05
    
    u_0 = f_u(M)
    v_0 = f_v(M)
    
    #A = Neumann_mat(M)
    A = Dirichlet_mat(M)
    #print(A)
    u_i = u_0
    v_i = v_0

    t_eval = 100
    u_time_mat = np.zeros((M+1,M+1,t_eval))
    v_time_mat = np.zeros((M+1,M+1,t_eval))
    counter = 0
    
    for t in range(1,T+1):
        if t % 100 == 0:
            print('developing in time...' , t)
        #w_i = step_euler(w_i,tlist[t],dwdt)
        u_i, v_i = forward_euler(r_u,r_v,u_i,v_i,A,f,k,h)
        if t % int(T/t_eval) == 0:
            u_mat = np.reshape(u_i,(M+1,M+1))
            v_mat = np.reshape(v_i,(M+1,M+1))
            u_time_mat[:,:,counter] = u_mat
            v_time_mat[:,:,counter] = v_mat
            counter +=1
    
    #animering under

    print('Animating, sit tight...')

    fig, (ax0,ax1) = plt.subplots(1,2,figsize=(30, 12))
    ax0.set(xlim=(0, 1), ylim=(0, 1))
    ax1.set(xlim=(0, 1), ylim=(0, 1))
    ax0.set_title('U(t)',fontsize = 18)
    ax1.set_title('V(t)',fontsize = 18)
    
    cax0 = ax0.pcolormesh(x, y, u_time_mat[:-1, :-1, 0], vmin=0, vmax=1)
    fig.colorbar(cax0, ax = ax0)
    
    cax1 = ax1.pcolormesh(x, y, v_time_mat[:-1, :-1, 0],vmin=0, vmax=1)
    fig.colorbar(cax1, ax = ax1)

    def animate(i):
        cax0.set_array(u_time_mat[:-1, :-1, i].flatten())
        cax1.set_array(v_time_mat[:-1, :-1, i].flatten()) 
    
 
    anim = FuncAnimation(fig, animate, interval = 100, frames=t_eval)
    anim.save('animation.gif', writer='pillow')
    
    plt.draw()   
    plt.show(anim)
    '''
    fig1, (ax2,ax3) = plt.subplots(1,2,figsize=(30, 12))
    ax2.set(xlim=(0, 1), ylim=(0, 1))
    ax3.set(xlim=(0, 1), ylim=(0, 1))
    ax2.set_title('U(t)',fontsize = 18)
    ax3.set_title('V(t)',fontsize = 18)
    
    
    cax2 = ax2.pcolor(x, y, u_time_mat[:,:,-1], vmin=0, vmax=1)
    fig1.colorbar(cax2, ax = ax2)
    
    cax3 = ax3.pcolor(x, y, v_time_mat[:, :,-1],vmin=0, vmax=1)
    fig1.colorbar(cax3, ax = ax3)
    '''    
main()

print('Program has ended, hopefully without issues..')