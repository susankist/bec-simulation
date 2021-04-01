# -*- coding: utf-8 -*-
"""
Created on Thu Oct 8 09:45:14 2020

@author: Gordon Robb
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
    f_psi1_k = open('psi1_k.out',"w") #mom1
    f_psi1_k.close()
    f_psi2_k = open('psi2_k.out',"w") #mom2
    f_psi2_k.close()
    f_expect_mom = open('expect_mom.out',"w") #mom
    f_expect_mom.close()

#Read input data from file
def readinput(): 
    fname0='patt1d_q_sfm_fft_s.in'
    data0 = np.genfromtxt(fname0,comments='!')#load data from input file
    #skips one line at end of file and ignores everything on a line after "!"

    nodes=data0[0].astype(int) #no. of points in each direction first input is called nodes and assigned type integer     
    maxt=data0[1] #time of interaction
    #maxt=maxt*2
    ht=data0[2] #timestep
    width_psi=data0[3] #width of wavefunction in x
    #width_psi=0.1
    p0=data0[4] #pump saturation parameter 1.9e-9
    p0=1.9e-9
    Delta=data0[5] 
    #Delta=-Delta
    gambar=data0[6] #Gamma/omega_r
    b0=data0[7] #optical density
    q_crit=data0[8] #critical wavenumber same as 'n' in paper?
    R=data0[9] #mirror reflectivity
    gbar=data0[10] #scattering length parameter
    plotnum=data0[11].astype(int) #no. of outputs
    width_psi2=data0[14] #2nd wave width
    #width_psi2=width_psi
    mom1=data0[13].astype(int)
    mom1 = 0
    mom2=data0[15].astype(int)
    mom2 = 0
    v0 = -3 #momentum optical field
    return nodes,maxt,ht,width_psi,p0,Delta,gambar,b0,q_crit,R,gbar,plotnum,width_psi2,mom1,mom2,v0 #added 2nd wave width_psi2

#Initialise variables 
def initvars():
    shift=np.pi/2.0/q_crit**2 #theta
    q_filter=q_crit*3 #anything above this is set to zero after fft to cancel out other instability curves
    #q_filter = 25
    hx=2.0*np.pi/np.float32(nodes) #dimensionless spatial step
    tperplot=maxt/np.float32(plotnum-1) #time per plot
    x=np.linspace(0,2.0*np.pi-hx,nodes)-np.pi #x from 0-pi to (2pi-hx)-pi with no. of element = nodes
    
    psi1=np.complex64(np.exp(-(x-0)**2/(2.0*width_psi**2))*np.exp(1j*mom1*(x-0))) #initial gaussian wave (x-2.5) for 2.5 offset
    norm=hx*np.sum(np.abs(psi1)**2) #normalisation constant 
    psi1=psi1/np.sqrt(norm)*np.sqrt(2.0*np.pi) #normalise y0
    
    psi2=np.complex64(np.exp(-(x)**2/(2.0*width_psi2**2))*np.exp(1j*mom2*(x))) #2nd wave 
    norm2=hx*np.sum(np.abs(psi2)**2)
    psi2=psi2/np.sqrt(norm2)*np.sqrt(2.0*np.pi)
    
    y0=np.concatenate((psi1,psi2)) #!
    noise=np.random.random_sample(nodes)*1.0e-4 #random numbers (no. of points=nodes)
    kx=np.fft.fftfreq(nodes, d=hx)*2.0*np.pi #related to n in paper
    #kx = wavenum. discrete fourier transform sample freq. with window length=nodes and sample spacing=hx
    
    return shift,q_filter,hx,tperplot,x,y0,noise,kx

#Write data to output files
def output(t,y):
    psi1=y[0:nodes] #input y now called psi within 'output' #added y2 2nd wave
    psi2=y[nodes:2*nodes]
    psi_sq=0.5*(np.abs(psi1)**2+np.abs(psi2)**2)
    F=np.sqrt(p0)*np.exp(-1j*b0/(2.0*Delta)*psi_sq)*(np.ones(nodes)+noise)*np.exp(1j*v0*x) #forward field F where sqrt(p0) = F(0,t).get by integrating eq. 6 wrt z
    Fk=np.fft.fft(F) #1D discrete Fourier transform
    Fk=np.where(np.abs(kx)>q_filter,0.0*1j,Fk) #Fk set to 0 (cmplx) for abs(kx)>q_filter and set to Fk otherwise
    F=np.fft.ifft(Fk) #inverse Fourier transform
    B=calc_B(F,shift) #backward field
    s=np.abs(F)**2+np.abs(B)**2 #intensity
    error=hx*np.sum(psi_sq)-2.0*np.pi 
    mod=np.max(s)-np.min(s) #modulus of s
    #save intensity data
    f_s = open('s.out',"a+") #opens file to append
    data=np.concatenate(([t],s))#joins arrays t and s into one flattened array listing first t and then s in same line 
    np.savetxt(f_s,data.reshape((1,nodes+1)), fmt='%1.3E',delimiter=' ')
    #save array 'data' into textfile called f_s, with format %1.3E? and -space- between  coloumns
    f_s.close()
    
    #save BEC density data
    #added factor of 0.5 12/11/20!!
    f_psi = open('psi.out',"a+") #same as above now done with [t] and abs(psi)^2 into file 'psi.out'
    data=np.concatenate(([t],0.5*(np.abs(psi1)**2+np.abs(psi2)**2)))
    np.savetxt(f_psi,data.reshape((1,nodes+1)), fmt='%1.3E',delimiter=' ')
    f_psi.close()
    
    f_psi1 = open('psi1.out',"a+")
    data=np.concatenate(([t],np.abs(psi1)**2))
    np.savetxt(f_psi1,data.reshape((1,nodes+1)),fmt='%1.3E',delimiter=' ')
    f_psi1.close()
    
    f_psi2 = open('psi2.out',"a+")
    data=np.concatenate(([t],np.abs(psi2)**2))
    np.savetxt(f_psi2,data.reshape((1,nodes+1)),fmt='%1.3E',delimiter=' ')
    f_psi2.close()
    
    psi1_k,mom1_expect=mom_dist(psi1)   
    psi2_k,mom2_expect=mom_dist(psi2)
    
    f_psi1_k = open('psi1_k.out',"a+")
    data=np.concatenate(([t],np.abs(psi1_k)**2))
    np.savetxt(f_psi1_k,data.reshape((1,nodes+1)), fmt='%1.3E',delimiter=' ')
    f_psi1_k.close()

    f_psi2_k = open('psi2_k.out',"a+")
    data=np.concatenate(([t],np.abs(psi2_k)**2))
    np.savetxt(f_psi2_k,data.reshape((1,nodes+1)), fmt='%1.3E',delimiter=' ')
    f_psi2_k.close()
    
    f_expect_mom = open('expect_mom.out',"a+")
    data=np.concatenate(([t],[mom1_expect],[mom2_expect]))
    np.savetxt(f_expect_mom,data.reshape((1,3)), fmt='%1.3E')
    f_expect_mom.close()
    
    progress=np.int(t/maxt*100) #calculate percentage of progressed time
    print('Completed '+str(progress)+' % :  mod = '+str(mod)+',  Error ='+str(error))
    #print progressed time %, modulus of s (s=intensity), error  

    return t,mod,error


def mom_dist(psi):
    psi_k = np.fft.fft(psi)
    k_expect=np.sum(np.abs(psi_k)**2*kx)/np.sum(np.abs(psi_k)**2)
    psi_k = psi_k/(nodes)*np.sqrt(2*np.pi)
    psi_k=np.fft.fftshift(psi_k)
    
    return psi_k,k_expect 
    

#Integrate kinetic energy part of Schrodinger equation
def propagate_bec(y,tstep):
    psi1=y[0:nodes]
    psi1_k=np.fft.fft(psi1) #fourier transform psi
    psi1_k=psi1_k*np.exp(-1j/gambar*kx**2*tstep) #multiply by phase factor from KE part of SE 
    psi1=np.fft.ifft(psi1_k) #1D inverse discrete fourier transform
    
    psi2=y[nodes:2*nodes]
    psi2_k=np.fft.fft(psi2) #fourier transform psi
    psi2_k=psi2_k*np.exp(-1j/gambar*kx**2*tstep) #multiply by phase factor from KE part of SE 
    psi2=np.fft.ifft(psi2_k) #1D inverse discrete fourier transform
    
    newy = np.concatenate((psi1,psi2))
    
    return newy

#Propagate optical field in free space to calculate backward field (B)
def calc_B(F,shift): 
    Fk=np.fft.fft(F) #fourier transform of forward optical field
    Bk=np.sqrt(R)*Fk*np.exp(-1j*kx**2*shift) #calculate backwards optical field with mirror refl. and phase factor: Eq. 8 
    B=np.fft.ifft(Bk) #inverse discrete fourier transform to get B
    
    return B

#2nd order Runge-Kutta algorithm                  , solves nonlinear ODE dy/dt -> y(t+dt)    
def rk2(t,y):
    yk1=ht*dy(t,y) 
    tt=t+0.5*ht
    yt=y+0.5*yk1
    yk2=ht*dy(tt,yt)
        
    newt=t+ht
    newy=y+yk2 #y(t+dt) = y(t) + dt(F(y(t))) + dt/2*F(y(t))
    
    return newt,newy


#RHS of ODEs for integration of potential energy part of Schrodinger equation KE=0
def dy(t,y):
    psi1=y[0:nodes]
    psi2=y[nodes:2*nodes]
    psi_sq=0.5*(np.abs(psi1)**2+np.abs(psi2)**2) #incoherent
    F=np.sqrt(p0)*np.exp(-1j*b0/(2.0*Delta)*psi_sq)*(np.ones(nodes)+noise)*np.exp(1j*v0*x) #F = integral of Eq. 6 wrt z 
    Fk=np.fft.fft(F)
    Fk=np.where(np.abs(kx)>q_filter,0.0*1j,Fk) #set Fk equal to Fk for abs(kx)<0, otherwise 0.
    F=np.fft.ifft(Fk)
    B=calc_B(F,shift)
    dy1=-1j*Delta/4.0*(np.abs(F)**2+np.abs(B)**2)*psi1  
    dy2=-1j*Delta/4.0*(np.abs(F)**2+np.abs(B)**2)*psi2
    dy=np.concatenate((dy1,dy2))  
    
    return dy #dy = RHS of SE with KE = 0

##########
openfiles()
nodes,maxt,ht,width_psi,p0,Delta,gambar,b0,q_crit,R,gbar,plotnum,width_psi2,mom1,mom2,v0=readinput()
shift,q_filter,hx,tperplot,x,y0,noise,kx=initvars()
y=y0
t=0.0
nextt=tperplot
ind=0 
n=0
output(t,y) #psi=y. F calculated, FFT(F)->filter->IFFT(FFT(F)). B calculated. s calculated. data for s and |psi|^2 saved 
while (t<maxt): #until t reaches maxt
    y=propagate_bec(y,0.5*ht) #kin E #psi=y. FFT(psi) -> phase factor:half time step -> IFFT(FFT(psi))
    t,y=rk2(t,y) #pot E setting t and y to newt(= t + timestep ht) and newy (=y(t+dt))-solved ODE of pot. energy part of SE
    y=propagate_bec(y,0.5*ht)
    if (t>=nextt): 
        n=n+1
        output(t,y) #now using newy after BEC propagation and newt after timestep
        ind=ind+1
        nextt=nextt+tperplot 

#for plots of momentum crossing times:
fname1 = "s.out"
fname7 = "expect_mom.out"
data1 = np.loadtxt(fname1) 
data7 = np.loadtxt(fname7)
t = data1[:,0] 
mom1_expect=data7[:,1] 
mom2_expect=data7[:,2]


deltap = mom1_expect-mom2_expect
deltap_abs = abs(deltap)
delta_magp = abs(mom1_expect) - abs(mom2_expect)
ind=np.where(t == 5.0e9) 
deltap_secondhalf = deltap[len(deltap)//2:] 
dif_initial = mom1_expect[0] - mom2_expect[0] #difference in intial momenta

#for finding max delta p between t = 5e9 and t = 10e9:
deltap_abs_secondhalf = abs(deltap_secondhalf)
max_deltap_secondhalf = np.amax(deltap_abs_secondhalf) #maximum momentum difference

#std(deltap) between t = 5e9 and t = 10e9
std_deltap_secondhalf = np.std(deltap_secondhalf)

first_idx = np.argmax(deltap_abs<0.4) #return index where deltap_abs is < 0.3
#first_idx = np.argmax(mom1_expect<0)-1 #return index before p1 turns negative
first_t = t[first_idx] #time of intersection
inter_pump = p0 #pump power
first_1 = mom1_expect[first_idx] #momentum of first BEC at intersection
first_2 = mom2_expect[first_idx] #should be same as above
deltap_t = mom1_expect[ind] - mom2_expect[ind] #difference in momenta at t=5e9
#note: mom1 and mom2 are initial momenta
delta_magp_afterintersect =  np.delete(delta_magp,np.arange(0,first_idx,1))
std_delta_magp_afterintersect = np.std(delta_magp_afterintersect)

meanv1 = np.mean(mom1_expect)
meanv2 = np.mean(mom2_expect)
gapv = first_1 - first_2

f_intersect = open('intersect.out',"a+") #opens file to append
data=np.array([first_t,p0,deltap_abs[first_idx],dif_initial,deltap_t,first_1,first_2,mom1,mom2, max_deltap_secondhalf, std_deltap_secondhalf, std_delta_magp_afterintersect,meanv1,meanv2,gapv])#joins arrays t and s into one flattened array listing first t and then s in same line 
data=data.T
np.savetxt(f_intersect,data.reshape(1,15), fmt='%1.3E',delimiter=' ')
#save array 'data' into textfile called f_s, with format %1.3E? and -space- between  coloumns
f_intersect.close()

#for plots to find threshold when changing q_filter
fname2 = 'psi.out'
data2 = np.loadtxt(fname2)
prob= data2[:,1:]
max_mag = np.amax(prob) #maximum value of total BEC magnitude squared (incoherent)
min_mag = np.amin(prob)
dif_mag = max_mag - min_mag

f_filter = open('filter.out','a+')
data=np.array([p0,mom1,mom2,q_filter,dif_mag])
np.savetxt(f_filter,data.reshape(1,5), fmt='%1.3E',delimiter=' ')
f_filter.close()

fig=plt.figure(figsize=(6,5))

plt.plot(t,delta_magp)
plt.xlabel("$t$", fontsize=17)
plt.ylabel("$|p_1|-|p_2|$", fontsize=17)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()

print('Finished.')
