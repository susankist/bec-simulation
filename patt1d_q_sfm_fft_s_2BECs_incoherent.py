# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:11:33 2021

@author: Gordon Robb (adapted by Susan Kistenberger)
"""
import numpy as np
import matplotlib.pyplot as plt

#Open new output data files
def openfiles():                   
    f_s = open('s.out',"w") #opens (or creates if doesn't exist) file called "s.out" for writing
    f_s.close() #closes 's.out' file
    f_psi = open('psi.out',"w")
    f_psi.close()
    f_psi1 = open('psi1.out',"w")
    f_psi1.close()
    f_psi2 = open('psi2.out',"w")
    f_psi2.close()
    f_psi1_k = open('psi1_k.out',"w") 
    f_psi1_k.close()
    f_psi2_k = open('psi2_k.out',"w") 
    f_psi2_k.close()
    f_expect_mom = open('expect_mom.out',"w") 
    f_expect_mom.close()

#Read input data from file
def readinput(): 
    fname0='patt1d_q_sfm_fft_s.in'
    data0 = np.genfromtxt(fname0,comments='!')#load data from input file
    #skips one line at end of file and ignores anything on a line after "!"
    nodes=data0[0].astype(int) #no. of points in each direction     
    maxt=data0[1] #time of interaction
    ht=data0[2] #timestep
    width_psi=data0[3] #width of wavefunction in x
    width_psi2=data0[14] #2nd wave width (introduced by S.K.)
    p0=data0[4] #pump saturation parameter
    Delta=data0[5] #associated with field-atom detuning
    gambar=data0[6] #Gamma/omega_r 
    b0=data0[7] #optical density
    q_crit=data0[8] #critical wavenumber
    R=data0[9] #mirror reflectivity
    gbar=data0[10] #scattering length parameter
    plotnum=data0[11].astype(int) #no. of outputs
    
    #introduced by S.K.: momentum components
    mom1=data0[13].astype(int) #momentum BEC1
    mom2=data0[15].astype(int) #momentum BEC2
    v0 = 0 #momentum optical field
    
    return nodes,maxt,ht,width_psi,p0,Delta,gambar,b0,q_crit,R,gbar,plotnum,width_psi2,mom1,mom2,v0 #added 2nd wave width_psi2

#Initialise variables 
def initvars():
    shift=np.pi/2.0/q_crit**2 #phase shift due to free space propagation
    q_filter=q_crit*3 #anything above this is set to zero after fft to cancel out other instability curves
    hx=2.0*np.pi/np.float32(nodes) #dimensionless spatial step
    tperplot=maxt/np.float32(plotnum-1) #time per plot
    x=np.linspace(0,2.0*np.pi-hx,nodes)-np.pi #x from 0-pi to (2pi-hx)-pi with no. of element = nodes
    
    #adapted by S.K.: introduction of 2nd BEC and momentum components in matter waves
    psi1=np.complex64(np.exp(-(x-0)**2/(2.0*width_psi**2))*np.exp(1j*mom1*(x-0))) #initial gaussian matter wave 1 centred at x = 0
    norm=hx*np.sum(np.abs(psi1)**2) #normalisation constant 
    psi1=psi1/np.sqrt(norm)*np.sqrt(2.0*np.pi) #normalise matter wave
    
    psi2=np.complex64(np.exp(-(x)**2/(2.0*width_psi2**2))*np.exp(1j*mom2*(x))) #initial gaussian matter wave 2 centred at x = 0
    norm2=hx*np.sum(np.abs(psi2)**2)
    psi2=psi2/np.sqrt(norm2)*np.sqrt(2.0*np.pi)
    
    y0=np.concatenate((psi1,psi2))
    noise=np.random.random_sample(nodes)*1.0e-4 #random numbers (no. of points = nodes)
    kx=np.fft.fftfreq(nodes, d=hx)*2.0*np.pi 
    #kx = wavenum. discrete fourier transform sample freq. with window length=nodes and sample spacing=hx
    
    return shift,q_filter,hx,tperplot,x,y0,noise,kx

#Write data to output files
def output(t,y):
    psi1=y[0:nodes] #selection of 1st matter wave 
    psi2=y[nodes:2*nodes] #... 2nd matter wave
    psi_sq=0.5*(np.abs(psi1)**2+np.abs(psi2)**2) #incoherent addition of matter waves -> two-component BECs
    F=np.sqrt(p0)*np.exp(-1j*b0/(2.0*Delta)*psi_sq)*(np.ones(nodes)+noise)*np.exp(1j*v0*x) #forward optical field
    Fk=np.fft.fft(F) #1D discrete Fourier transform
    Fk=np.where(np.abs(kx)>q_filter,0.0*1j,Fk) #Fk set to 0 for abs(kx)>q_filter and set to Fk otherwise -> include only 1st instability curve
    F=np.fft.ifft(Fk) #inverse Fourier transform
    B=calc_B(F,shift) #backward optical field
    s=np.abs(F)**2+np.abs(B)**2 #optical intensity
    error=hx*np.sum(psi_sq)-2.0*np.pi 
    mod=np.max(s)-np.min(s) #modulus of s
    
    #save intensity data
    f_s = open('s.out',"a+") #opens file to append
    data=np.concatenate(([t],s)) #joins arrays t and s into one flattened array listing first t and then s in same line 
    np.savetxt(f_s,data.reshape((1,nodes+1)), fmt='%1.3E',delimiter=' ') #save array 'data' into textfile called f_s
    f_s.close()
    
    #save two-component BEC density data
    f_psi = open('psi.out',"a+")
    data=np.concatenate(([t],0.5*(np.abs(psi1)**2+np.abs(psi2)**2)))
    np.savetxt(f_psi,data.reshape((1,nodes+1)), fmt='%1.3E',delimiter=' ')
    f_psi.close()
    
    #save BEC1 density data
    f_psi1 = open('psi1.out',"a+")
    data=np.concatenate(([t],np.abs(psi1)**2))
    np.savetxt(f_psi1,data.reshape((1,nodes+1)),fmt='%1.3E',delimiter=' ')
    f_psi1.close()
    
    #save BEC2 density data
    f_psi2 = open('psi2.out',"a+")
    data=np.concatenate(([t],np.abs(psi2)**2))
    np.savetxt(f_psi2,data.reshape((1,nodes+1)),fmt='%1.3E',delimiter=' ')
    f_psi2.close()
    
    psi1_k,mom1_expect=mom_dist(psi1)   
    psi2_k,mom2_expect=mom_dist(psi2)
    
    #save BEC1 in momentum space data
    f_psi1_k = open('psi1_k.out',"a+")
    data=np.concatenate(([t],np.abs(psi1_k)**2))
    np.savetxt(f_psi1_k,data.reshape((1,nodes+1)), fmt='%1.3E',delimiter=' ')
    f_psi1_k.close()

    #save BEC2 in momentum space data
    f_psi2_k = open('psi2_k.out',"a+")
    data=np.concatenate(([t],np.abs(psi2_k)**2))
    np.savetxt(f_psi2_k,data.reshape((1,nodes+1)), fmt='%1.3E',delimiter=' ')
    f_psi2_k.close()
    
    #save expectation value of momentum data
    f_expect_mom = open('expect_mom.out',"a+")
    data=np.concatenate(([t],[mom1_expect],[mom2_expect]))
    np.savetxt(f_expect_mom,data.reshape((1,3)), fmt='%1.3E')
    f_expect_mom.close()
    
    progress=np.int(t/maxt*100) #calculate percentage of progressed time
    print('Completed '+str(progress)+' % :  mod = '+str(mod)+',  Error ='+str(error))
    #print progressed time %, modulus of s, error  

    return t,mod,error

#introduced by S.K.: 
#Convert matter waves to momentum space and determine expectation value of momentum
def mom_dist(psi):
    psi_k = np.fft.fft(psi) #1D discrete Fourier transform
    k_expect=np.sum(np.abs(psi_k)**2*kx)/np.sum(np.abs(psi_k)**2) #momentum expectation value
    psi_k = psi_k/(nodes)*np.sqrt(2*np.pi) #normalise matter wave
    psi_k=np.fft.fftshift(psi_k) #shift zero frequency component to middle of spectrum
    
    return psi_k,k_expect 
    
#Integrate kinetic energy part of Schrödinger equation
def propagate_bec(y,tstep):
    psi1=y[0:nodes] #selection of 1st matter wave
    psi1_k=np.fft.fft(psi1) #1D discrete Fourier transform
    psi1_k=psi1_k*np.exp(-1j/gambar*kx**2*tstep) #multiply by phase factor from KE part of SE 
    psi1=np.fft.ifft(psi1_k) #1D inverse discrete fourier transform
    
    #introduced by S.K.: repeat process for 2nd matter wave
    psi2=y[nodes:2*nodes] #selection of 2nd matter wave
    psi2_k=np.fft.fft(psi2) #1D discrete Fourier transform
    psi2_k=psi2_k*np.exp(-1j/gambar*kx**2*tstep) #multiply by phase factor from KE part of SE 
    psi2=np.fft.ifft(psi2_k) #1D inverse discrete fourier transform
    
    newy = np.concatenate((psi1,psi2))
    
    return newy

#Propagate optical field in free space to calculate backward field (B)
def calc_B(F,shift): 
    Fk=np.fft.fft(F) #Fourier transform of forward optical field
    Bk=np.sqrt(R)*Fk*np.exp(-1j*kx**2*shift) #calculate backward field with mirror reflectivity and phase factor
    B=np.fft.ifft(Bk) #1D inverse discrete Fourier transform
    
    return B

#2nd order Runge-Kutta algorithm, solves nonlinear ODE dy/dt -> y(t+dt): Solve potential energy part of Schrödinger equation    
def rk2(t,y):
    yk1=ht*dy(t,y) #ht = full time step
    tt=t+0.5*ht
    yt=y+0.5*yk1
    yk2=ht*dy(tt,yt)
        
    newt=t+ht
    newy=y+yk2 #y(t+dt) = y(t) + dt(F(y(t))) + dt/2*F(y(t))
    
    return newt,newy


#For integration of potential energy part of Schrödinger equation (used in rk2)
def dy(t,y):
    psi1=y[0:nodes] #select 1st matter wave
    psi2=y[nodes:2*nodes] #select 2nd matter wave
    psi_sq=0.5*(np.abs(psi1)**2+np.abs(psi2)**2) #incoherent addition of matter waves (two-component BECs)
    F=np.sqrt(p0)*np.exp(-1j*b0/(2.0*Delta)*psi_sq)*(np.ones(nodes)+noise)*np.exp(1j*v0*x) #forward optical field
    Fk=np.fft.fft(F) #1D discrete Fourier transform
    Fk=np.where(np.abs(kx)>q_filter,0.0*1j,Fk) #use q_filter to include only first instability curve
    F=np.fft.ifft(Fk) #1D inverse discrete Fourier transform
    B=calc_B(F,shift) #obtain backward field
    
    #RHS of Schrödinger equation (kinetic energy = 0)
    dy1=-1j*Delta/4.0*(np.abs(F)**2+np.abs(B)**2)*psi1 #...for BEC1
    dy2=-1j*Delta/4.0*(np.abs(F)**2+np.abs(B)**2)*psi2 #...for BEC2 (added by S.K.)
    dy=np.concatenate((dy1,dy2))  
    
    return dy #RHS of Schrödinger equation with KE = 0

##########
openfiles()
nodes,maxt,ht,width_psi,p0,Delta,gambar,b0,q_crit,R,gbar,plotnum,width_psi2,mom1,mom2,v0=readinput()
shift,q_filter,hx,tperplot,x,y0,noise,kx=initvars()
y=y0
t=0.0
nextt=tperplot
ind=0 
output(t,y) #generate initial output
while (t<maxt): #until t reaches maxt
    y=propagate_bec(y,0.5*ht) #solve kinetic energy part of Schrödinger equation (half time step)
    t,y=rk2(t,y) #solve potential energy part of Schrödinger equation (full time step)
    y=propagate_bec(y,0.5*ht) #solve kinetic energy part of Schrödinger equation (half time step)
    if (t>=nextt): 
        output(t,y) #now using newy after BEC propagation and newt after timestep
        ind=ind+1 
        nextt=nextt+tperplot 

print('Finished.')
