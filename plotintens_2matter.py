# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:13:11 2020

@author: Gordon Robb (adapted by Susan Kistenberger)
"""

#Plots output from code PART1D_Q_SFM_FFT_S_2BEC_INCOHERENT.PY 
#Shows image of optical intensity and BEC density vs x and t
#Shows momentum evolution of BEC1 and BEC2 vs v and t

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 500 #set figure dpi
plt.rcParams["font.size"] = "17" #set figure font size

#fname = raw_input("Enter filename: ")
plt.rcParams['ps.usedistiller']='xpdf'                            # improves quality of .eps figures for use with LaTeX

fname1="psi1.out" 
fname2="psi2.out"
fname3="s.out"
fname4="psi1_k.out" 
fname5="psi2_k.out"
fname6="expect_mom.out"
fname7="psi.out"

data1 = np.loadtxt(fname1)                                       #load dataset in the form t, |Psi1(x,t)|^2
data2 = np.loadtxt(fname2)                                       #load dataset in the form t, |Psi2(x,t)|^2
data3 = np.loadtxt(fname3)                                       #load dataset in the form t, intensity
data4 = np.loadtxt(fname4)                                       #load dataset in the form t, |Psi1(k,t)|^2
data5 = np.loadtxt(fname5)                                       #load dataset in the form t, |Psi2(k,t)|^2
data6 = np.loadtxt(fname6)
data7 = np.loadtxt(fname7)                                       #load dataset in the form t, |Psi(x,t)|^2

Nx=np.sqrt(np.size(data1,axis=1)-1).astype(int)                  #No. of points in each row for field and BEC

tvec = data1[:,0]  #time data
plotnum=len(tvec) #number of data points
psi1=data1[:,1:] #BEC1 density in spatial scale 
psi2=data2[:,1:] #BEC2 density in spatial scale 
s=data3[:,1:] #optical intensity
psi1_k=data4[:,1:] #BEC1 density in momentum scale
psi2_k=data5[:,1:] #BEC2 density in momentum scale
mom1_expect=data6[:,1] #expectation value of momentum of BEC1
mom2_expect=data6[:,2] #expectation value of momentum of BEC2

pi=4.0*np.arctan(1.0) 
xco=np.linspace(-pi,pi,Nx) #spatial coordinates

Nx=np.size(data1,axis=1)-1 
kco=list(range(int(-Nx/2),int(Nx/2)+1,1)) #momentum window
mini=int(Nx/2-10) #lower momentum window limit
maxi=int(Nx/2+10) #upper momentum window limit
psi1_k=psi1_k[:,mini:maxi+1] #select values to only plot central region of spectrum
psi2_k=psi2_k[:,mini:maxi+1] 

fig=plt.figure(figsize=(8,12))

#Atomic density in spatial scale x and time t of BEC1
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
f1=ax1.imshow(psi1, extent=[-pi,pi,0,tvec.max()], origin="lower", vmin=psi1.min(),vmax=psi1.max(), aspect="auto", cmap='bone')
cb1=fig.colorbar(f1,orientation='horizontal')
ax1.set_xlabel('$x$', fontsize=20)
ax1.set_ylabel('$t$', fontsize=20)
ax1.set_title(r'  $|\psi_1(x,t)|^2$',fontsize=20)

#Atomic density in spatial scale x and time t of BEC2
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
f2=ax2.imshow(psi2, extent=[-pi,pi,0,tvec.max()], origin="lower", vmin=psi2.min(),vmax=psi2.max(), aspect="auto", cmap='bone')
cb2=fig.colorbar(f2,orientation='horizontal')
ax2.set_xlabel('$x$', fontsize=20)
ax2.set_yticklabels([])
ax2.set_title(r'$|\psi_2(x,t)|^2$',fontsize=20)

#Optical intensity in spatial scale x and time t
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
f3=ax3.imshow(s, extent=[-pi,pi,0,tvec.max()], origin="lower", vmin=s.min(),vmax=s.max(), aspect="auto", cmap='bone')
cb3=fig.colorbar(f3,orientation='horizontal')
ax3.set_xlabel('$x$', fontsize=20)
ax3.set_yticklabels([])
ax3.set_title('$s(x,t)$',fontsize=20)

#Momentum evolution in time t of BEC1
ax4 = plt.subplot2grid((2,6), (1,0), colspan=2)
f4=ax4.imshow(psi1_k, extent=[kco[mini],kco[maxi],0,tvec.max()], origin="lower", vmin=psi1_k.min(),vmax=psi1_k.max(), aspect="auto", cmap='bone')
cb4=fig.colorbar(f4,orientation='horizontal')
ax4.set_xlabel('$v_1$', fontsize=20)
ax4.set_ylabel('$t$', fontsize=20)
ax4.set_title(r'   $|\psi_1(v_1,t)|^2$',fontsize=20)

#Momentum evolution in time t of BEC2
ax5 = plt.subplot2grid((2,6), (1,4), colspan=2)
f5=ax5.imshow(psi2_k, extent=[kco[mini],kco[maxi],0,tvec.max()], origin="lower", vmin=psi2_k.min(),vmax=psi2_k.max(), aspect="auto",cmap='bone')
cb5=fig.colorbar(f5,orientation='horizontal')
ax5.set_xlabel('$v_2$', fontsize=20)
ax5.set_yticklabels([])
ax5.set_title(r'$|\psi_2(v_2,t)|^2$',fontsize=20)

plt.show()

  



