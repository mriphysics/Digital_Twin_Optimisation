#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:04:27 2023

@author: dw16
"""

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import csv
from matplotlib.ticker import MultipleLocator

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray."""
    return x.detach().cpu().numpy()

#%% FIGURE 3: Plotting all waveforms for unconstrained optimzations.

[V_SEC,G_SEC] = torch.load('SEC3000_TWs.pt')
[V_LEC,G_LEC] = torch.load('UC_LEC_PAPERv3_TWs.pt')
[V_BEC,G_BEC] = torch.load('UC_BEC_PAPERv3_TWs.pt')
[V_0S,G_0S]     = torch.load('SEC3000_oTWs.pt')
[V_0L,G_0L]     = torch.load('UC_LEC_PAPERv3_oTWs.pt')
[V_0B,G_0B]     = torch.load('UC_BEC_PAPERv3_oTWs.pt')

coarse_tstep = 0.0001
tsamples     = 100
no_TR        = 0 # NDummies

t_axis = torch.linspace(0,coarse_tstep*(torch.Tensor.size(V_SEC,0)-1),torch.Tensor.size(V_SEC,0))

idx1 = tsamples*no_TR
idx2 = tsamples*(no_TR+5)

plt.figure(2)
plt.subplot(221)
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*V_0B[idx1:idx2].cpu().detach(),color = 'dimgray',linestyle="--")
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*G_0S[idx1:idx2].cpu().detach(),color = 'lightcoral')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*G_0L[idx1:idx2].cpu().detach(),color = 'limegreen')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*G_0B[idx1:idx2].cpu().detach(),color = 'cornflowerblue')
plt.axvspan(2.5, 5.6, facecolor='grey', alpha=0.15)
plt.xlabel('Time [ms]',fontsize=20,fontweight='bold')
plt.ylabel('x Gradient [mT/m]',fontsize=20,fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)
plt.xlim(0,10)
plt.ylim(-46,19)
plt.legend(['$g_{0}$','$G_{0}^{short}$','$G_{0}^{long}$','$G_{0}^{both}$'],fontsize=24,loc='lower right',ncol=2)
plt.text(-1,16,'(A)',fontsize=28)

plt.subplot(222)
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*V_SEC[idx1:idx2].cpu().detach(),color = 'lightcoral',linestyle="--")
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*G_SEC[idx1:idx2].cpu().detach(),color = 'lightcoral')
plt.axvspan(2.5, 5.6, facecolor='grey', alpha=0.15)
plt.annotate('', xy=(2.4,8.7), xytext=(1.7,8.7), arrowprops=dict(color='lightcoral', arrowstyle='->',linewidth=2))
plt.xlabel('Time [ms]',fontsize=20)
plt.ylabel('x Gradient [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)
plt.xlim(0,10)
plt.ylim(-46,19)
plt.legend(['$g_{op}^{short}$','$G_{op}^{short}$'],fontsize=24,loc='lower right',ncol=2)
plt.text(-1,16,'(B)',fontsize=28)

plt.subplot(223)
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*V_LEC[idx1:idx2].cpu().detach(),color = 'limegreen',linestyle="--")
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*G_LEC[idx1:idx2].cpu().detach(),color = 'limegreen')
plt.axvspan(2.5, 5.6, facecolor='grey', alpha=0.15)
plt.xlabel('Time [ms]',fontsize=20)
plt.ylabel('x Gradient [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)
plt.xlim(0,10)
plt.ylim(-46,19)
plt.legend(['$g_{op}^{long}$','$G_{op}^{long}$'],fontsize=24,loc='lower right',ncol=2)
plt.text(-1,16,'(C)',fontsize=28)

plt.subplot(224)
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*V_BEC[idx1:idx2].cpu().detach(),color = 'cornflowerblue',linestyle="--")
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*G_BEC[idx1:idx2].cpu().detach(),color = 'cornflowerblue')
plt.axvspan(2.5, 5.6, facecolor='grey', alpha=0.15)
plt.xlabel('Time [ms]',fontsize=20)
plt.ylabel('x Gradient [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)
plt.xlim(0,10)
plt.ylim(-46,19)
plt.legend(['$g_{op}^{both}$','$G_{op}^{both}$'],fontsize=24,loc='lower right',ncol=2)
plt.text(-1,16,'(D)',fontsize=28)

plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.1,hspace=0.2)

#%% FIGURE 4: Final plot for UC BEC optimization.

coarse_tstep = 0.0001

file = open('UC_BEC_PAPERv3.txt') 

PDG_data1           = torch.load('UC_BEC_PAPERv31.pth') #CHANGE THIS!
PDG_data2           = torch.load('UC_BEC_PAPERv32.pth') #CHANGE THIS!
[tw_cat3x,tw_cat4x] = torch.load('UC_BEC_PAPERv3_TWs.pt') #CHANGE THIS!

gradc = PDG_data1.get('gradc')
slewc = PDG_data1.get('slewc')

klocs_perturbed = PDG_data1.get('klocs_perturbed')
klocs_target    = PDG_data1.get('klocs_target')
klocs_opt       = PDG_data2.get('klocs_opt')
reco_target1    = PDG_data1.get('reco_target')
reco_perturb  = PDG_data1.get('reco_perturb')
reco_target2  = PDG_data2.get('reco_target')
reco_opt      = PDG_data2.get('reco_opt')

loss_history = PDG_data2.get('loss_history')

slew3x = (tw_cat3x[1:] - tw_cat3x[:-1]) / coarse_tstep
slew4x = (tw_cat4x[1:] - tw_cat4x[:-1]) / coarse_tstep

fig = plt.figure(1)
gs  = fig.add_gridspec(2,6)

plot1 = fig.add_subplot(gs[0,:2])
plt.plot(klocs_target[:,0].cpu().detach().numpy(), klocs_target[:,1].cpu().detach().numpy(), 'k.', label='TARGET')
plt.title('$k_{0}$',fontsize=26,fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.ylim([-20,20])
plt.xlim([-20,20])
plot1.set_ylabel('$k_y$',fontsize=20)
plot1.set_xlabel('$k_x$',fontsize=20)
plt.text(-29.5,21,'(A)',fontsize=28)
plt.text( 20.5,21,'(B)',fontsize=28)
plt.text( 70.2,21,'(C)',fontsize=28)
plt.text(-23,-33,'(D)',fontsize=28)
plt.text( 27,-33,'(E)',fontsize=28)
plt.text( 77,-33,'(F)',fontsize=28)

plot2 = fig.add_subplot(gs[0,2:4])
plt.plot(klocs_perturbed[:,0].cpu().detach().numpy(), klocs_perturbed[:,1].cpu().detach().numpy(), 'k.', label='PERTURBED')
plt.title('$\widetilde{k}$', fontsize= 26,fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.ylim([-20,20])
plt.xlim([-20,20])
plot2.set_ylabel('$k_y$',fontsize=20)
plot2.set_xlabel('$k_x$',fontsize=20)

plot3 = fig.add_subplot(gs[0,4:])
plt.plot(klocs_opt[:,0].cpu().detach().numpy(), klocs_opt[:,1].cpu().detach().numpy(), 'k.', label='PERTURBED')
plt.title('$k_{op}$',fontsize=26,fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.ylim([-20,20])
plt.xlim([-20,20])
plot3.set_ylabel('$k_y$',fontsize=20)
plot3.set_xlabel('$k_x$',fontsize=20)

max_sig = torch.max(reco_target1)

ax4 = fig.add_subplot(gs[1,:2])
plot4 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_target1/max_sig)))))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
cbar = plt.colorbar(plot4,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_xticks([])
ax4.set_yticks([])
plt.title('$I_{0}$',fontsize=26,fontweight='bold')

ax5 = fig.add_subplot(gs[1,2:4])
plot5 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((reco_target1/max_sig))))-np.rot90(to_numpy(torch.abs((reco_perturb/max_sig)))))))
cbar = plt.colorbar(plot5,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax5.set_xticklabels([])
ax5.set_yticklabels([])
ax5.set_xticks([])
ax5.set_yticks([])
plt.title('$I_{0}-\widetilde{I}$',fontsize=26,fontweight='bold')

ax6 = fig.add_subplot(gs[1,4:])
plot6 = plt.imshow(np.abs(np.rot90(to_numpy(torch.abs((reco_target1/max_sig))))-np.rot90(to_numpy(torch.abs((reco_opt/max_sig))))))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
cbar = plt.colorbar(plot6,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.set_xticks([])
ax6.set_yticks([])
plt.title('$I_{0}-I_{op}$',fontsize=26,fontweight='bold')

plt.subplots_adjust(top=0.95, bottom=0.05, left=0.07, right=0.92,wspace=0.65,hspace=0.35)

t_axis = torch.linspace(0,coarse_tstep*(torch.Tensor.size(tw_cat3x,0)-1),torch.Tensor.size(tw_cat3x,0))

tsamples = 100
no_TR    = 0 # NDummies

idx1 = tsamples*no_TR
idx2 = tsamples*(no_TR+5)

csvreader = csv.reader(file)
rows = []
for row in csvreader:
    rows.append(row)
file.close()
del rows[1::2] # Remove odd rows.

row1 = np.zeros(len(rows))
row2 = np.zeros(len(rows))
row3 = np.zeros(len(rows))
row4 = np.zeros(len(rows))
row5 = np.zeros(len(rows))

for pp in range(0,len(rows)):
    row1[pp] = rows[pp][0]
    row2[pp] = rows[pp][1]
    row3[pp] = rows[pp][2]
    row4[pp] = rows[pp][3]
    row5[pp] = rows[pp][4]

plt.figure(2)
plt.subplot(222)
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*tw_cat3x[idx1:idx2].cpu().detach(),color = 'cornflowerblue',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*tw_cat4x[idx1:idx2].cpu().detach(),color = 'cornflowerblue')
plt.axvspan(2.5, 5.6, facecolor='grey', alpha=0.15)
plt.axvspan(2.5+10, 5.6+10, facecolor='grey', alpha=0.15)
plt.axvspan(2.5+20, 5.6+20, facecolor='grey', alpha=0.15)
plt.axvspan(2.5+30, 5.6+30, facecolor='grey', alpha=0.15)
plt.axvspan(2.5+40, 5.6+40, facecolor='grey', alpha=0.15)

plt.xlabel('Time [ms]',fontsize=20)
plt.ylabel('x Gradient [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)
plt.xlim(0,50), plt.ylim(-45,20)
plt.legend(['$g_{op}$','$G_{op}$'],fontsize=24,loc='lower right',ncol=2)
plt.text(-68,20,'(G)',fontsize=28)
plt.text( -8,20,'(H)',fontsize=28)
plt.annotate('', xy=( 9,-1), xytext=( 9,-8), arrowprops=dict(color='cornflowerblue', arrowstyle='->',linewidth=2))
plt.annotate('', xy=(19,-1), xytext=(19,-8), arrowprops=dict(color='cornflowerblue', arrowstyle='->',linewidth=2))
plt.annotate('', xy=(29,-1), xytext=(29,-8), arrowprops=dict(color='cornflowerblue', arrowstyle='->',linewidth=2))
plt.annotate('', xy=(39,-1), xytext=(39,-8), arrowprops=dict(color='cornflowerblue', arrowstyle='->',linewidth=2))
plt.annotate('', xy=(49,-1), xytext=(49,-8), arrowprops=dict(color='cornflowerblue', arrowstyle='->',linewidth=2))

plt.subplot(221)
plt.plot(row1,linewidth=2,color='black')
plt.plot(row2,linewidth=2,color='gray')
plt.plot(row5,linewidth=2,color='gainsboro')
plt.xlabel('Iteration',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.yscale('log')
plt.legend(['$L_{I}$','$L_{k}$','$L_{tRF}$'],fontsize=24,loc='upper right')

plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.1,hspace=0.2)

#%% FIGURE 5: Pre-emphasis comparison.

plt.style.use('default')

coarse_tstep_pe = 1e-5

[Nsamples_10,default_10,waveform_10,Gmaxline_10] = torch.load('PEdata_10mTm.pt')
[Nsamples_15,default_15,waveform_15,Gmaxline_15] = torch.load('PEdata_15mTm.pt')
[Nsamples_20,default_20,waveform_20,Gmaxline_20] = torch.load('PEdata_20mTm.pt')

Nsamples_default = 1000
t_axis = torch.linspace(0,coarse_tstep_pe*(32*Nsamples_default-1),32*Nsamples_default)

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].add_patch(Rectangle((0, -10), 20, 20,facecolor='plum',alpha=0.1))
ax[0,0].add_patch(Rectangle((0, -15), 20,  5,facecolor='teal',alpha=0.1))
ax[0,0].add_patch(Rectangle((0,  15), 20, -5,facecolor='teal',alpha=0.1))
ax[0,0].add_patch(Rectangle((0, -20), 20,  5,facecolor='orange',alpha=0.1))
ax[0,0].add_patch(Rectangle((0,  20), 20, -5,facecolor='orange',alpha=0.1))

TR_idx = 16
ax[0,0].plot(t_axis[0:Nsamples_default]*1000,default_15[TR_idx*Nsamples_default:(TR_idx+1)*Nsamples_default].cpu().detach().numpy()*1000,'dimgray',linewidth=2)
ax[0,0].plot(t_axis[0:Nsamples_10]*1000,waveform_10[TR_idx*Nsamples_10:(TR_idx+1)*Nsamples_10].cpu().detach().numpy()*1000,'plum',linewidth=2)
ax[0,0].plot(t_axis[0:Nsamples_15]*1000,waveform_15[TR_idx*Nsamples_15:(TR_idx+1)*Nsamples_15].cpu().detach().numpy()*1000,'teal',linewidth=2)
ax[0,0].plot(t_axis[0:Nsamples_20]*1000,waveform_20[TR_idx*Nsamples_20:(TR_idx+1)*Nsamples_20].cpu().detach().numpy()*1000,'orange',linewidth=2)
ax[0,0].legend(['_1','_2','_3','_4','_5','$g_{0}$','10mT/m','15mT/m','20mT/m'],fontsize=20,loc='lower right',ncol=2)

ax[0,0].grid()
ax[0,0].tick_params(axis='x', labelsize=18)
ax[0,0].tick_params(axis='y', labelsize=18)
ax[0,0].set_xlabel('Time [ms]',fontsize=20)
ax[0,0].set_ylabel('x Gradient [mT/m]',fontsize=20)
ax[0,0].xaxis.set_minor_locator(MultipleLocator(1))
ax[0,0].xaxis.set_major_locator(MultipleLocator(2))
ax[0,0].grid(which='major',axis='x',linewidth=2)
ax[0,0].grid(which='minor',axis='x',linewidth=0.5)
ax[0,0].grid(which='major',axis='y',linewidth=1)
ax[0,0].grid(which='minor',axis='y',linewidth=1)
ax[0,0].set_xlim(0,10)
ax[0,0].set_title('Pre-emphasis with variable TE',fontsize=24,fontweight='bold')
plt.text(-23.5,122,'(A)',fontsize=28)
plt.text(  5.4,122,'(B)',fontsize=28)
plt.text(-23.5,51,'(C)',fontsize=28)
plt.text(  4.4,51,'(D)',fontsize=28)

Gmaxp = [10.1,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
TEp   = [4.21,4.08,3.96,3.85,3.76,3.68,3.61,3.55,3.5,3.45,3.4,3.37,3.33,3.3,3.27,3.24,3.22,3.19,3.17,3.15,3.13,3.12,3.1]

GmaxM  = [10,15,20,25,30]
Gmaxp2 = [10,15,20,25,30,32] 

Errp  = [9.31,6.63,6.19,2.76,2.76,0.002]
ErrM  = [3.825,2.8633,2.2064,0.7211,0.7245]

PFp   = [49.22,34.38,26.37,9.18,9.18,0]
PFM   = [14.26,6.64,3.125,0,0]

ax[0,1].plot(Gmaxp,TEp, '-o',color='black',linewidth=2,markersize=10)
ax[0,1].set_xlabel('$g_{max}$ [mT/m]',fontsize=20)
ax[0,1].set_ylabel('TE [ms]',fontsize=20)
ax[0,1].tick_params(labelsize=18)
ax[0,1].tick_params(axis='x',labelsize=18)
ax[0,1].tick_params(axis='y',labelsize=18)
ax[0,1].xaxis.set_minor_locator(MultipleLocator(2.5))
ax[0,1].xaxis.set_major_locator(MultipleLocator(5))
ax[0,1].grid(which='major',axis='x',linewidth=2)
ax[0,1].grid(which='minor',axis='x',linewidth=0.5)
ax[0,1].grid(which='major',axis='y',linewidth=1)
ax[0,1].grid(which='minor',axis='y',linewidth=1)
ax[0,1].set_xlim(9,33)
ax[0,1].set_ylim(3.05,4.3)
ax[0,1].legend(['Pre-emphasis with variable TE'],fontsize=20,loc='upper right') #

ax[1,0].plot(Gmaxp2,Errp, '-o',color='black',linewidth=2,markersize=10)
ax[1,0].plot(GmaxM,ErrM, '-o',color='gray',linewidth=2,markersize=10)
ax[1,0].set_xlabel('$g_{max}$ [mT/m]',fontsize=20)
ax[1,0].set_ylabel('NRMSE [%]',fontsize=20)
ax[1,0].tick_params(labelsize=18)
ax[1,0].tick_params(axis='x',labelsize=18)
ax[1,0].tick_params(axis='y',labelsize=18)
ax[1,0].xaxis.set_minor_locator(MultipleLocator(2.5))
ax[1,0].xaxis.set_major_locator(MultipleLocator(5))
ax[1,0].grid(which='major',axis='x',linewidth=2)
ax[1,0].grid(which='minor',axis='x',linewidth=0.5)
ax[1,0].grid(which='major',axis='y',linewidth=1)
ax[1,0].grid(which='minor',axis='y',linewidth=1)
ax[1,0].set_xlim(9,33)
ax[1,0].legend(['Pre-emphasis with constant TE','Digital Twin Optimization'],fontsize=20,loc='upper right')

ax[1,1].plot(Gmaxp2,PFp, '-o',color='black',linewidth=2,markersize=10)
ax[1,1].plot(GmaxM,PFM, '-o',color='gray',linewidth=2,markersize=10)
ax[1,1].set_xlabel('$g_{max}$ [mT/m]',fontsize=20)
ax[1,1].set_ylabel('Skipped k-space samples [%]',fontsize=20)
ax[1,1].tick_params(labelsize=18)
ax[1,1].tick_params(axis='x',labelsize=18)
ax[1,1].tick_params(axis='y',labelsize=18)
ax[1,1].xaxis.set_minor_locator(MultipleLocator(2.5))
ax[1,1].xaxis.set_major_locator(MultipleLocator(5))
ax[1,1].grid(which='major',axis='x',linewidth=2)
ax[1,1].grid(which='minor',axis='x',linewidth=0.5)
ax[1,1].grid(which='major',axis='y',linewidth=1)
ax[1,1].grid(which='minor',axis='y',linewidth=1)
ax[1,1].set_xlim(9,33)
ax[1,1].legend(['Pre-emphasis with constant TE','Digital Twin Optimization'],fontsize=20,loc='upper right')

fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.1,hspace=0.3)

#%% FIGURE 6: Individual amplitude and slew rate constrained results.

coarse_tstep = 0.0001
TR_idx = torch.linspace(0,31,32).int()

AC0_data = torch.load('UC_BEC10_PAPERv32.pth')
AC0_reco = AC0_data.get('reco_opt')
AC0_kloc = AC0_data.get('klocs_opt')
AC0_data2 = torch.load('UC_BEC10_PAPERv3_TWs.pt')
AC0tw3 = AC0_data2[0]
AC0tw4 = AC0_data2[1]
AC0_data3 = torch.load('UC_BEC10_PAPERv3_yTWs.pt')
AC0tw3y = AC0_data3[0]
AC0tw4y = AC0_data3[1]

# OVERWRITE RESULTS WITH PF-CORRECTED RESULTS
AC0_data3 = torch.load('UC_BEC10_PAPERv3_PFresults.pt')
AC0_kloc = AC0_data3[0]
AC0_reco = AC0_data3[1]

AC1_data = torch.load('UC_BEC15_PAPERv32.pth')
AC1_reco = AC1_data.get('reco_opt')
AC1_kloc = AC1_data.get('klocs_opt')
AC1_data2 = torch.load('UC_BEC15_PAPERv3_TWs.pt')
AC1tw3 = AC1_data2[0]
AC1tw4 = AC1_data2[1]
AC1_data3 = torch.load('UC_BEC15_PAPERv3_yTWs.pt')
AC1tw3y = AC1_data3[0]
AC1tw4y = AC1_data3[1]

# OVERWRITE RESULTS WITH PF-CORRECTED RESULTS
AC1_data3 = torch.load('UC_BEC15_PAPERv3_PFresults.pt')
AC1_kloc = AC1_data3[0]
AC1_reco = AC1_data3[1]

AC2_data = torch.load('UC_BEC20_PAPERv32.pth')
AC2_reco = AC2_data.get('reco_opt')
AC2_kloc = AC2_data.get('klocs_opt')
AC2_data2 = torch.load('UC_BEC20_PAPERv3_TWs.pt')
AC2tw3 = AC2_data2[0]
AC2tw4 = AC2_data2[1]
AC2_data3 = torch.load('UC_BEC20_PAPERv3_yTWs.pt')
AC2tw3y = AC2_data3[0]
AC2tw4y = AC2_data3[1]

# OVERWRITE RESULTS WITH PF-CORRECTED RESULTS
AC2_data3 = torch.load('UC_BEC20_PAPERv3_PFresults.pt')
AC2_kloc = AC2_data3[0]
AC2_reco = AC2_data3[1]

AC_data = torch.load('UC_BEC20_PAPERv31.pth')
target = AC_data.get('reco_target')
max_sig = torch.max(target)

t_axis = torch.linspace(0,coarse_tstep*(torch.Tensor.size(AC2tw3,0)-1),torch.Tensor.size(AC2tw4,0))

tsamples = 100
no_TR    = 0
idx1 = tsamples*(no_TR)
idx2 = tsamples*(no_TR+5)

fig = plt.figure(1)
gs  = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[1,2])
plot1 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((target/max_sig))))-np.rot90(to_numpy(torch.abs((AC0_reco/max_sig)))))))
cbar = plt.colorbar(plot1,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,0.1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xticks([])
ax1.set_yticks([])
plt.title(' ',fontsize=18,fontweight='bold')

ax2 = fig.add_subplot(gs[1,1])
plot2 = plt.imshow(np.abs(np.rot90(to_numpy(torch.abs((AC0_reco/max_sig))))))
cbar = plt.colorbar(plot2,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])
plt.title(' ',fontsize=18,fontweight='bold')

ax3 = fig.add_subplot(gs[1,0])
plt.plot(AC0_kloc[:,0].cpu().detach().numpy(), AC0_kloc[:,1].cpu().detach().numpy(), 'k.',markersize=8)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.ylim([-20,20])
plt.xlim([-20,20])
ax3.set_ylabel('$k_y$',fontsize=20)
ax3.set_xlabel('$k_x$',fontsize=20)
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.gca().grid(which='major',axis='x',linewidth=1)
plt.gca().grid(which='minor',axis='x',linewidth=1)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

ax4 = fig.add_subplot(gs[0,:])
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC0tw3[idx1:idx2].cpu().detach(),color = 'cornflowerblue',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC0tw4[idx1:idx2].cpu().detach(),color = 'cornflowerblue')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC0tw3y[idx1:idx2].cpu().detach(),color = 'orchid',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC0tw4y[idx1:idx2].cpu().detach(),color = 'orchid')
plt.axvspan(3+0, 5.6+0, facecolor='grey', alpha=0.15)
plt.axvspan(3+10, 5.6+10, facecolor='grey', alpha=0.15)
plt.axvspan(3+20, 5.6+20, facecolor='grey', alpha=0.15)
plt.axvspan(3+30, 5.6+30, facecolor='grey', alpha=0.15)
plt.axvspan(3+40, 5.6+40, facecolor='grey', alpha=0.15)
plt.annotate('', xy=(5.5,-9), xytext=(4.2,-9), arrowprops=dict(color='black', arrowstyle='->',linewidth=2))
plt.annotate('', xy=(15.5,-9), xytext=(14.2,-9), arrowprops=dict(color='black', arrowstyle='->',linewidth=2))
plt.annotate('', xy=(25.5,-9), xytext=(24.2,-9), arrowprops=dict(color='black', arrowstyle='->',linewidth=2))
plt.annotate('', xy=(35.5,-9), xytext=(34.2,-9), arrowprops=dict(color='black', arrowstyle='->',linewidth=2))
plt.annotate('', xy=(45.5,-9), xytext=(44.2,-9), arrowprops=dict(color='black', arrowstyle='->',linewidth=2))

plt.xlabel('Time [ms]',fontsize=20,horizontalalignment='right',x=1.0)
plt.ylabel('Gradient [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(['$g_{x,op}$','$G_{x,op}$','$g_{y,op}$','$G_{y,op}$'],fontsize=24,loc='upper right',ncol=4)
plt.xlim(0,50)
plt.ylim(-25,25)
plt.text(-3,  25, '(A)',fontsize=28)
plt.text(0.2, -42.2, '(B)',fontsize=28)
plt.text(21.1,-42.5, '(C)',fontsize=28,color='w')
plt.text(37.1,-42.5, '(D)',fontsize=28,color='w')

plt.text(2, 27, '$g_{max}$=10mT/m',fontsize=26)
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

plt.tight_layout()
fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.09,wspace=-0.1,hspace=0.25)

idx1 = tsamples*(no_TR)
idx2 = tsamples*(no_TR+5)

fig = plt.figure(2)
gs  = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[1,2])
plot1 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((target/max_sig))))-np.rot90(to_numpy(torch.abs((AC1_reco/max_sig)))))))
cbar = plt.colorbar(plot1,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,0.1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xticks([])
ax1.set_yticks([])
plt.title(' ',fontsize=18,fontweight='bold')

ax2 = fig.add_subplot(gs[1,1])
plot2 = plt.imshow(np.abs(np.rot90(to_numpy(torch.abs((AC1_reco/max_sig))))))
cbar = plt.colorbar(plot2,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])
plt.title(' ',fontsize=18,fontweight='bold')

ax3 = fig.add_subplot(gs[1,0])
plt.plot(AC1_kloc[:,0].cpu().detach().numpy(), AC1_kloc[:,1].cpu().detach().numpy(), 'k.',markersize=8)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.ylim([-20,20])
plt.xlim([-20,20])
ax3.set_ylabel('$k_y$',fontsize=20)
ax3.set_xlabel('$k_x$',fontsize=20)
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.gca().grid(which='major',axis='x',linewidth=1)
plt.gca().grid(which='minor',axis='x',linewidth=1)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

ax4 = fig.add_subplot(gs[0,:])
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC1tw3[idx1:idx2].cpu().detach(),color = 'cornflowerblue',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC1tw4[idx1:idx2].cpu().detach(),color = 'cornflowerblue')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC1tw3y[idx1:idx2].cpu().detach(),color = 'orchid',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC1tw4y[idx1:idx2].cpu().detach(),color = 'orchid')
plt.axvspan(2.8+0, 5.6+0, facecolor='grey', alpha=0.15)
plt.axvspan(2.8+10, 5.6+10, facecolor='grey', alpha=0.15)
plt.axvspan(2.8+20, 5.6+20, facecolor='grey', alpha=0.15)
plt.axvspan(2.8+30, 5.6+30, facecolor='grey', alpha=0.15)
plt.axvspan(2.8+40, 5.6+40, facecolor='grey', alpha=0.15)
plt.xlabel('Time [ms]',fontsize=20,horizontalalignment='right',x=1.0)
plt.ylabel('Gradient [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(['$g_{x,op}$','$G_{x,op}$','$g_{y,op}$','$G_{y,op}$'],fontsize=24,loc='upper right',ncol=4)
plt.xlim(0,50)
plt.ylim(-25,25)
plt.text(-3,  25, '(E)',fontsize=28)
plt.text(0.2, -42.2, '(F)',fontsize=28)
plt.text(21.1,-42.5, '(G)',fontsize=28,color='w')
plt.text(37.1,-42.5, '(H)',fontsize=28,color='w')

plt.text(2, 27, '$g_{max}$=15mT/m',fontsize=26)
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

plt.tight_layout()
fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.09,wspace=-0.1,hspace=0.25)

idx1 = tsamples*(no_TR)
idx2 = tsamples*(no_TR+5)

fig = plt.figure(3)
gs  = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[1,2])
plot1 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((target/max_sig))))-np.rot90(to_numpy(torch.abs((AC2_reco/max_sig)))))))
cbar = plt.colorbar(plot1,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,0.1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xticks([])
ax1.set_yticks([])
plt.title(' ',fontsize=18,fontweight='bold')

ax2 = fig.add_subplot(gs[1,1])
plot2 = plt.imshow(np.abs(np.rot90(to_numpy(torch.abs((AC2_reco/max_sig))))))
cbar = plt.colorbar(plot2,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])
plt.title(' ',fontsize=18,fontweight='bold')

ax3 = fig.add_subplot(gs[1,0])
plt.plot(AC2_kloc[:,0].cpu().detach().numpy(), AC2_kloc[:,1].cpu().detach().numpy(), 'k.',markersize=8)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.ylim([-20,20])
plt.xlim([-20,20])
ax3.set_ylabel('$k_y$',fontsize=20)
ax3.set_xlabel('$k_x$',fontsize=20)
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.gca().grid(which='major',axis='x',linewidth=1)
plt.gca().grid(which='minor',axis='x',linewidth=1)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

ax4 = fig.add_subplot(gs[0,:])
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC2tw3[idx1:idx2].cpu().detach(),color = 'cornflowerblue',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC2tw4[idx1:idx2].cpu().detach(),color = 'cornflowerblue')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC2tw3y[idx1:idx2].cpu().detach(),color = 'orchid',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC2tw4y[idx1:idx2].cpu().detach(),color = 'orchid')
plt.axvspan(2.6+0, 5.6+0, facecolor='grey', alpha=0.15)
plt.axvspan(2.6+10, 5.6+10, facecolor='grey', alpha=0.15)
plt.axvspan(2.6+20, 5.6+20, facecolor='grey', alpha=0.15)
plt.axvspan(2.6+30, 5.6+30, facecolor='grey', alpha=0.15)
plt.axvspan(2.6+40, 5.6+40, facecolor='grey', alpha=0.15)
plt.xlabel('Time [ms]',fontsize=20,horizontalalignment='right',x=1.0)
plt.ylabel('Gradient [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(['$g_{x,op}$','$G_{x,op}$','$g_{y,op}$','$G_{y,op}$'],fontsize=24,loc='upper right',ncol=4)
plt.xlim(0,50)
plt.ylim(-25,25)
plt.text(-3,  25, '(I)',fontsize=28)
plt.text(0.2, -42.2, '(J)',fontsize=28)
plt.text(21.1,-42.5, '(K)',fontsize=28,color='w')
plt.text(37.1,-42.5, '(L)',fontsize=28,color='w')

plt.text(2, 27, '$g_{max}$=20mT/m',fontsize=26)
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

plt.tight_layout()
fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.09,wspace=-0.1,hspace=0.25)

#%% FIGURE 7: All fully-constrained results.

coarse_tstep = 0.0001
TR_idx = torch.linspace(0,31,32).int()

AC0_data = torch.load('C_BEC_PAPERv32.pth')
AC0_reco = AC0_data.get('reco_opt')
AC0_kloc = AC0_data.get('klocs_opt')
AC0_data2 = torch.load('C_BEC_PAPERv3_TWs.pt')
AC0tw3 = AC0_data2[0]
AC0tw4 = AC0_data2[1]
AC0_data3 = torch.load('C_BEC_PAPERv3_yTWs.pt')
AC0tw3y = AC0_data3[0]
AC0tw4y = AC0_data3[1]

# OVERWRITE RESULTS WITH PF-CORRECTED RESULTS
AC0_data3 = torch.load('C_BEC_PAPERv3_PFresults.pt')
AC0_kloc = AC0_data3[0]
AC0_reco = AC0_data3[1]

AC1_data = torch.load('C_LEC_PAPERv32.pth')
AC1_reco = AC1_data.get('reco_opt')
AC1_kloc = AC1_data.get('klocs_opt')
AC1_data2 = torch.load('C_LEC_PAPERv3_TWs.pt')
AC1tw3 = AC1_data2[0]
AC1tw4 = AC1_data2[1]
AC1_data3 = torch.load('C_LEC_PAPERv3_yTWs.pt')
AC1tw3y = AC1_data3[0]
AC1tw4y = AC1_data3[1]

# OVERWRITE RESULTS WITH PF-CORRECTED RESULTS
AC1_data3 = torch.load('C_LEC_PAPERv3_PFresults.pt')
AC1_kloc = AC1_data3[0]
AC1_reco = AC1_data3[1]

AC2_data = torch.load('C_SEC_PAPERv32.pth')
AC2_reco = AC2_data.get('reco_opt')
AC2_kloc = AC2_data.get('klocs_opt')
AC2_data2 = torch.load('C_SEC_PAPERv3_TWs.pt')
AC2tw3 = AC2_data2[0]
AC2tw4 = AC2_data2[1]
AC2_data3 = torch.load('C_SEC_PAPERv3_yTWs.pt')
AC2tw3y = AC2_data3[0]
AC2tw4y = AC2_data3[1]

# OVERWRITE RESULTS WITH PF-CORRECTED RESULTS
AC2_data3 = torch.load('C_SEC_PAPERv3_PFresults.pt')
AC2_kloc = AC2_data3[0]
AC2_reco = AC2_data3[1]

AC_data = torch.load('C_BEC_PAPERv31.pth')
target = AC_data.get('reco_target')
max_sig = torch.max(target)

t_axis = torch.linspace(0,coarse_tstep*(torch.Tensor.size(AC2tw3,0)-1),torch.Tensor.size(AC2tw4,0))

tsamples = 100
no_TR    = 0
idx1 = tsamples*(no_TR)
idx2 = tsamples*(no_TR+5)

fig = plt.figure(3)
gs  = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[1,2])
plot1 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((target/max_sig))))-np.rot90(to_numpy(torch.abs((AC0_reco/max_sig)))))))
cbar = plt.colorbar(plot1,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,0.1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xticks([])
ax1.set_yticks([])
plt.title(' ',fontsize=18,fontweight='bold')

ax2 = fig.add_subplot(gs[1,1])
plot2 = plt.imshow(np.abs(np.rot90(to_numpy(torch.abs((AC0_reco/max_sig))))))
cbar = plt.colorbar(plot2,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])
plt.title(' ',fontsize=18,fontweight='bold')

ax3 = fig.add_subplot(gs[1,0])
plt.plot(AC0_kloc[:,0].cpu().detach().numpy(), AC0_kloc[:,1].cpu().detach().numpy(), 'k.',markersize=8)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.ylim([-20,20])
plt.xlim([-20,20])
ax3.set_ylabel('$k_y$',fontsize=20)
ax3.set_xlabel('$k_x$',fontsize=20)
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.gca().grid(which='major',axis='x',linewidth=1)
plt.gca().grid(which='minor',axis='x',linewidth=1)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

ax4 = fig.add_subplot(gs[0,:])
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC0tw3[idx1:idx2].cpu().detach(),color = 'cornflowerblue',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC0tw4[idx1:idx2].cpu().detach(),color = 'cornflowerblue')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC0tw3y[idx1:idx2].cpu().detach(),color = 'orchid',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC0tw4y[idx1:idx2].cpu().detach(),color = 'orchid')
plt.axvspan(2.9+0, 5.6+0, facecolor='grey', alpha=0.15)
plt.axvspan(2.9+10, 5.6+10, facecolor='grey', alpha=0.15)
plt.axvspan(2.9+20, 5.6+20, facecolor='grey', alpha=0.15)
plt.axvspan(2.9+30, 5.6+30, facecolor='grey', alpha=0.15)
plt.axvspan(2.9+40, 5.6+40, facecolor='grey', alpha=0.15)
plt.xlabel('Time [ms]',fontsize=20,horizontalalignment='right',x=1.0)
plt.ylabel('x Gradient [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(['$g_{x,op}$','$G_{x,op}$','$g_{y,op}$','$G_{y,op}$'],fontsize=24,loc='upper right',ncol=4)
plt.xlim(0,50)
plt.ylim(-25,25)
plt.text(-3,  25, '(I)',fontsize=28)
plt.text(0.2, -42.2, '(J)',fontsize=28)
plt.text(21.1,-42.5, '(K)',fontsize=28,color='w')
plt.text(37.1,-42.5, '(L)',fontsize=28,color='w')

plt.text(2, 27, r'Both ($\tau$ = 1ms & $\tau$ = 100ms)',fontsize=26)
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

plt.tight_layout()
fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.09,wspace=-0.1,hspace=0.25)

idx1 = tsamples*(no_TR)
idx2 = tsamples*(no_TR+5)

fig = plt.figure(2)
gs  = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[1,2])
plot1 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((target/max_sig))))-np.rot90(to_numpy(torch.abs((AC1_reco/max_sig)))))))
cbar = plt.colorbar(plot1,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,0.1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xticks([])
ax1.set_yticks([])
plt.title(' ',fontsize=18,fontweight='bold')

ax2 = fig.add_subplot(gs[1,1])
plot2 = plt.imshow(np.abs(np.rot90(to_numpy(torch.abs((AC1_reco/max_sig))))))
cbar = plt.colorbar(plot2,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])
plt.title(' ',fontsize=18,fontweight='bold')

ax3 = fig.add_subplot(gs[1,0])
plt.plot(AC1_kloc[:,0].cpu().detach().numpy(), AC1_kloc[:,1].cpu().detach().numpy(), 'k.',markersize=8)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.ylim([-20,20])
plt.xlim([-20,20])
ax3.set_ylabel('$k_y$',fontsize=20)
ax3.set_xlabel('$k_x$',fontsize=20)
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.gca().grid(which='major',axis='x',linewidth=1)
plt.gca().grid(which='minor',axis='x',linewidth=1)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

ax4 = fig.add_subplot(gs[0,:])
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC1tw3[idx1:idx2].cpu().detach(),color = 'cornflowerblue',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC1tw4[idx1:idx2].cpu().detach(),color = 'cornflowerblue')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC1tw3y[idx1:idx2].cpu().detach(),color = 'orchid',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC1tw4y[idx1:idx2].cpu().detach(),color = 'orchid')
plt.axvspan(2.9+0, 5.6+0, facecolor='grey', alpha=0.15)
plt.axvspan(2.9+10, 5.6+10, facecolor='grey', alpha=0.15)
plt.axvspan(2.9+20, 5.6+20, facecolor='grey', alpha=0.15)
plt.axvspan(2.9+30, 5.6+30, facecolor='grey', alpha=0.15)
plt.axvspan(2.9+40, 5.6+40, facecolor='grey', alpha=0.15)
plt.xlabel('Time [ms]',fontsize=20,horizontalalignment='right',x=1.0)
plt.ylabel('x Gradient [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(['$g_{x,op}$','$G_{x,op}$','$g_{y,op}$','$G_{y,op}$'],fontsize=24,loc='upper right',ncol=4)
plt.xlim(0,50)
plt.ylim(-25,25)
plt.text(-3,  25, '(E)',fontsize=28)
plt.text(0.2, -42.2, '(F)',fontsize=28)
plt.text(21.1,-42.5, '(G)',fontsize=28,color='w')
plt.text(37.1,-42.5, '(H)',fontsize=28,color='w')

plt.text(2, 27, r'Long ($\tau$ = 100ms)',fontsize=26)
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

plt.tight_layout()
fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.09,wspace=-0.1,hspace=0.25)

idx1 = tsamples*(no_TR)
idx2 = tsamples*(no_TR+5)

fig = plt.figure(1)
gs  = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[1,2])
plot1 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((target/max_sig))))-np.rot90(to_numpy(torch.abs((AC2_reco/max_sig)))))))
cbar = plt.colorbar(plot1,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,0.1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xticks([])
ax1.set_yticks([])
plt.title(' ',fontsize=18,fontweight='bold')

ax2 = fig.add_subplot(gs[1,1])
plot2 = plt.imshow(np.abs(np.rot90(to_numpy(torch.abs((AC2_reco/max_sig))))))
cbar = plt.colorbar(plot2,fraction=0.046,pad=0.04)
cbar.ax.tick_params(labelsize=18)
plt.set_cmap('gray')
plt.clim(0,1)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(18)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])
plt.title(' ',fontsize=18,fontweight='bold')

ax3 = fig.add_subplot(gs[1,0])
plt.plot(AC2_kloc[:,0].cpu().detach().numpy(), AC2_kloc[:,1].cpu().detach().numpy(), 'k.',markersize=8)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.ylim([-20,20])
plt.xlim([-20,20])
ax3.set_ylabel('$k_y$',fontsize=20)
ax3.set_xlabel('$k_x$',fontsize=20)
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.gca().grid(which='major',axis='x',linewidth=1)
plt.gca().grid(which='minor',axis='x',linewidth=1)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

ax4 = fig.add_subplot(gs[0,:])
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC2tw3[idx1:idx2].cpu().detach(),color = 'cornflowerblue',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC2tw4[idx1:idx2].cpu().detach(),color = 'cornflowerblue')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC2tw3y[idx1:idx2].cpu().detach(),color = 'orchid',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*AC2tw4y[idx1:idx2].cpu().detach(),color = 'orchid')
plt.axvspan(2.8+0, 5.6+0, facecolor='grey', alpha=0.15)
plt.axvspan(2.8+10, 5.6+10, facecolor='grey', alpha=0.15)
plt.axvspan(2.8+20, 5.6+20, facecolor='grey', alpha=0.15)
plt.axvspan(2.8+30, 5.6+30, facecolor='grey', alpha=0.15)
plt.axvspan(2.8+40, 5.6+40, facecolor='grey', alpha=0.15)
plt.xlabel('Time [ms]',fontsize=20,horizontalalignment='right',x=1.0)
plt.ylabel('x Gradient [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(['$g_{x,op}$','$G_{x,op}$','$g_{y,op}$','$G_{y,op}$'],fontsize=24,loc='upper right',ncol=4)
plt.xlim(0,50)
plt.ylim(-25,25)
plt.text(-3,  25, '(A)',fontsize=28)
plt.text(0.2, -42.2, '(B)',fontsize=28)
plt.text(21.1,-42.5, '(C)',fontsize=28,color='w')
plt.text(37.1,-42.5, '(D)',fontsize=28,color='w')

plt.text(2, 27, r'Short ($\tau$ = 1ms)',fontsize=26)
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

plt.tight_layout()
fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.09,wspace=-0.1,hspace=0.25)

#%% FIGURE 8: Impact of data augmentation.

coarse_tstep = 0.0001
TR_idx = torch.linspace(0,31,32).int()

o_data = torch.load('UC_BEC_PAPERv32.pth')
o_reco = o_data.get('reco_opt')
o_kloc = o_data.get('klocs_opt')
o_data2 = torch.load('UC_BEC_PAPERv3_TWs.pt')
otw3 = o_data2[0]
otw4 = o_data2[1]
o_error = torch.load('DA_errorsHIGH_v3.pt')
o_error = np.concatenate(([o_error[-1]],o_error[:-1])) # Relocate last element to first.

lk_data = torch.load('mLOW_wk2.pth')
lk_reco = lk_data.get('reco_opt')
lk_kloc = lk_data.get('klocs_opt')
lk_data2 = torch.load('mLOW_wk2_TWs.pt')
lktw3 = lk_data2[0]
lktw4 = lk_data2[1]
lk_error = torch.load('DA_errorsLOW_v3.pt')
lk_error = np.concatenate(([lk_error[-1]],lk_error[:-1])) # Relocate last element to first.

DA_data = torch.load('mLOW_wkDA_v32.pth')
DA_reco = DA_data.get('reco_opt')
DA_kloc = DA_data.get('klocs_opt')
DA_data2 = torch.load('mLOW_wkDA2_v3_TWs.pt')
DAtw3 = DA_data2[0]
DAtw4 = DA_data2[1]
DA_error = torch.load('DA_errorsLOWDA_v3.pt')
DA_error = np.concatenate(([DA_error[-1]],DA_error[:-1])) # Relocate last element to first.

t_data = torch.load('UC_BEC_PAPERv31.pth')
target = t_data.get('reco_target')

t_datar = torch.load('UC_BEC_PAPERv32.pth')
targetr = t_datar.get('reco_target')

t_axis = torch.linspace(0,coarse_tstep*(torch.Tensor.size(DAtw3,0)-1),torch.Tensor.size(DAtw3,0))

tsamples = 100
no_TR    = 0
idx1 = tsamples*no_TR
idx2 = tsamples*(no_TR+5)

img_no = np.linspace(1,21,21)

fig = plt.figure(1)
gs  = fig.add_gridspec(3,12)

ax1 = fig.add_subplot(gs[:,:5])
plt.plot(img_no,o_error,'-o',linewidth=2,color='black')
plt.plot(img_no,lk_error,'-o',linewidth=2,color='darkgrey')
plt.plot(img_no,DA_error,'-o',linewidth=2,color='gainsboro',markeredgewidth=2, markeredgecolor='deeppink')
plt.plot(img_no[0],o_error[0],'o',color='deeppink')
plt.plot(img_no[0],lk_error[0],'o',color='deeppink')
plt.text(0.6,11.15,'(A)',fontsize=24)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().yaxis.set_major_locator(MultipleLocator(2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.gca().grid(which='major',axis='x',linewidth=0.5)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)
ax1.scatter(1,o_error[0],marker='o',color='red')
plt.xlim(0.5,21.5)
plt.ylim(0,11.6)
plt.xticks([1,3,5,7,9,11,13,15,17,19,21],['1','3','5','7','9','11','13','15','17','19','21'])
plt.xlabel('Target image no.',fontsize=18)
plt.ylabel('NRMSE [%]',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['High $w_{k}$','Low $w_{k}$','Low $w_{k}$ + aug'],fontsize=20,loc='center right', bbox_to_anchor=(1, 0.7),ncol=2)

ax2 = fig.add_subplot(gs[0,5:9])
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*otw3[idx1:idx2].cpu().detach(),color = 'cornflowerblue',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*otw4[idx1:idx2].cpu().detach(),color = 'cornflowerblue')
plt.axvspan(2.5+10, 5.6+10, facecolor='grey', alpha=0.15)
plt.axvspan(2.5+20, 5.6+20, facecolor='grey', alpha=0.15)
plt.axvspan(2.5+30, 5.6+30, facecolor='grey', alpha=0.15)
plt.xlabel('Time [ms]',fontsize=18)
plt.ylabel('x Gradient [mT/m]',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['$g_{op}$','$G_{op}$'],fontsize=20,loc='lower right',ncol=2)
plt.xlim(10,40)
plt.ylim(-35,30)
plt.text(10.2, 22, '(B)',fontsize=24)
plt.text(13.3, 22, 'High $w_{k}$',fontweight='bold',fontsize=22)
plt.gca().yaxis.set_major_locator(MultipleLocator(20))
plt.gca().yaxis.set_minor_locator(MultipleLocator(10))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

ax3 = fig.add_subplot(gs[1,5:9])
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*lktw3[idx1:idx2].cpu().detach(),color = 'cornflowerblue',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*lktw4[idx1:idx2].cpu().detach(),color = 'cornflowerblue')
plt.axvspan(2.5+10, 5.6+10, facecolor='grey', alpha=0.15)
plt.axvspan(2.5+20, 5.6+20, facecolor='grey', alpha=0.15)
plt.axvspan(2.5+30, 5.6+30, facecolor='grey', alpha=0.15)
plt.xlabel('Time [ms]',fontsize=18)
plt.ylabel('x Gradient [mT/m]',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.legend(['$g_{op}$','$G_{op}$'],fontsize=20,loc='lower right',ncol=2)
plt.xlim(10,40)
plt.ylim(-35,30)
plt.text(10.2, 22, '(D)',fontsize=24)
plt.text(13.3, 22, 'Low $w_{k}$',fontweight='bold',fontsize=22)
plt.gca().yaxis.set_major_locator(MultipleLocator(20))
plt.gca().yaxis.set_minor_locator(MultipleLocator(10))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

ax4 = fig.add_subplot(gs[2,5:9])
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*DAtw3[idx1:idx2].cpu().detach(),color = 'cornflowerblue',linestyle='--')
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*DAtw4[idx1:idx2].cpu().detach(),color = 'cornflowerblue')
plt.axvspan(2.5+10, 5.6+10, facecolor='grey', alpha=0.15)
plt.axvspan(2.5+20, 5.6+20, facecolor='grey', alpha=0.15)
plt.axvspan(2.5+30, 5.6+30, facecolor='grey', alpha=0.15)
plt.xlabel('Time [ms]',fontsize=18)
plt.ylabel('x Gradient [mT/m]',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.legend(['$g_{op}$','$G_{op}$'],fontsize=20,loc='lower right',ncol=2)
plt.xlim(10,40)
plt.ylim(-35,30)
plt.text(10.2, 22, '(F)',fontsize=24)
plt.text(13, 22, 'Low $w_{k}$ + aug',fontweight='bold',fontsize=22)
plt.gca().yaxis.set_major_locator(MultipleLocator(20))
plt.gca().yaxis.set_minor_locator(MultipleLocator(10))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

ax5 = fig.add_subplot(gs[0,9:])
plt.plot(o_kloc[32:128,0].cpu().detach().numpy(), o_kloc[32:128,1].cpu().detach().numpy(),'k.',markersize=8)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.ylim([-16,-11])
plt.xlim([-20,20])
plt.text(-19.8, -11.7, '(C)',fontsize=24)
ax5.set_ylabel('$k_y$',fontsize=18)
ax5.set_xlabel('$k_x$',fontsize=18)
plt.gca().yaxis.set_major_locator(MultipleLocator(2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.gca().grid(which='major',axis='x',linewidth=1)
plt.gca().grid(which='minor',axis='x',linewidth=1)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

ax6 = fig.add_subplot(gs[1,9:])
plt.plot(lk_kloc[32:128,0].cpu().detach().numpy(), lk_kloc[32:128,1].cpu().detach().numpy(),'k.',markersize=8)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.ylim([-16,-11])
plt.xlim([-20,20])
plt.text(-19.8, -11.7, '(E)',fontsize=24)
ax6.set_ylabel('$k_y$',fontsize=18)
ax6.set_xlabel('$k_x$',fontsize=18)
plt.gca().yaxis.set_major_locator(MultipleLocator(2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.gca().grid(which='major',axis='x',linewidth=1)
plt.gca().grid(which='minor',axis='x',linewidth=1)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

ax7 = fig.add_subplot(gs[2,9:])
plt.plot(DA_kloc[32:128,0].cpu().detach().numpy(), DA_kloc[32:128,1].cpu().detach().numpy(),'k.',markersize=8)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.ylim([-16,-11])
plt.xlim([-20,20])
plt.text(-19.8, -11.7, '(G)',fontsize=24)
ax7.set_ylabel('$k_y$',fontsize=18)
ax7.set_xlabel('$k_x$',fontsize=18)
plt.gca().yaxis.set_major_locator(MultipleLocator(2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.gca().grid(which='major',axis='x',linewidth=1)
plt.gca().grid(which='minor',axis='x',linewidth=1)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

fig.subplots_adjust(left=0.05, right=0.97, top=0.97, bottom=0.08,wspace=2.5,hspace=0.3)

#%% FIGURE S1: Impact of null gradient during RF.

coarse_tstep = 0.0001
TR_idx = torch.linspace(0,31,32).int()

UCz_data = torch.load('nonzeroWRF2.pth')
UCz_reco = UCz_data.get('reco_opt')
UCz_kloc = UCz_data.get('klocs_opt')
UCz_data2 = torch.load('nonzeroWRF_TWs.pt')
UCztw3 = UCz_data2[0]
UCztw4 = UCz_data2[1]

UCz2_data = torch.load('UC_BEC_PAPERv32.pth')
UCz2_reco = UCz2_data.get('reco_opt')
UCz2_kloc = UCz2_data.get('klocs_opt')
UCz2_data2 = torch.load('UC_BEC_PAPERv3_TWs.pt')
UCz2tw3 = UCz2_data2[0]
UCz2tw4 = UCz2_data2[1]

UCnz_data = torch.load('zeroWRF2.pth')
UCnz_reco = UCnz_data.get('reco_opt')
UCnz_kloc = UCnz_data.get('klocs_opt')
UCnz_data2 = torch.load('zeroWRF_TWs.pt')
UCnztw3 = UCnz_data2[0]
UCnztw4 = UCnz_data2[1]

BCz_data = torch.load('CnonzeroWRF2.pth')
BCz_reco = BCz_data.get('reco_opt')
BCz_kloc = BCz_data.get('klocs_opt')
BCz_data2 = torch.load('CnonzeroWRF_TWs.pt')
BCztw3 = BCz_data2[0]
BCztw4 = BCz_data2[1]

BCz2_data = torch.load('C_BEC_PAPERv32.pth')
BCz2_reco = BCz2_data.get('reco_opt')
BCz2_kloc = BCz2_data.get('klocs_opt')
BCz2_data2 = torch.load('C_BEC_PAPERv3_TWs.pt')
BCz2tw3 = BCz2_data2[0]
BCz2tw4 = BCz2_data2[1]

BCnz_data = torch.load('CzeroWRF2.pth')
BCnz_reco = BCnz_data.get('reco_opt')
BCnz_kloc = BCnz_data.get('klocs_opt')
BCnz_data2 = torch.load('CzeroWRF_TWs.pt')
BCnztw3 = BCnz_data2[0]
BCnztw4 = BCnz_data2[1]

t_axis = torch.linspace(0,coarse_tstep*(torch.Tensor.size(BCnztw3,0)-1),torch.Tensor.size(BCnztw3,0))

tsamples = 100
no_TR    = 0
idx1 = tsamples*no_TR
idx2 = tsamples*(no_TR+5)

fig = plt.figure(1)
gs  = fig.add_gridspec(2,2)

ax2 = fig.add_subplot(gs[0,:])
ax2.add_patch(Rectangle((0, -28), 2, 55,facecolor='black',alpha=0.1))
ax2.add_patch(Rectangle((10, -28), 2, 55,facecolor='black',alpha=0.1))
ax2.add_patch(Rectangle((20, -28), 2, 55,facecolor='black',alpha=0.1))
ax2.add_patch(Rectangle((30, -28), 2, 55,facecolor='black',alpha=0.1))
ax2.add_patch(Rectangle((40, -28), 2, 55,facecolor='black',alpha=0.1))
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*(UCztw4[idx1:idx2].cpu().detach()),color = 'skyblue',linestyle="-",linewidth=2)
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*(UCnztw4[idx1:idx2].cpu().detach()),color = 'steelblue',linestyle="-",linewidth=2)
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*(UCz2tw4[idx1:idx2].cpu().detach()),color = 'slategray',linestyle=":",linewidth=3)
plt.xlabel('Time [ms]',fontsize=20)
plt.ylabel('$G_{op}$ [mT/m]',fontsize=20)
plt.legend(['_1','_2','_3','_4','_5','$w_{tRF} = 10000$','$w_{tRF} = 0$','$w_{tRF} = 10000$ repeat'],fontsize=18,loc='upper right',ncol=3)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)
plt.xlim(0,50)
plt.ylim(-28,27)
plt.text(0.2, 22, '(A)',fontsize=26)
plt.text(2.2, 22, 'Unconstrained',fontweight='bold',fontsize=26)

ax5 = fig.add_subplot(gs[1,:])
ax5.add_patch(Rectangle((0, -35), 2, 60,facecolor='black',alpha=0.1))
ax5.add_patch(Rectangle((10, -35), 2, 60,facecolor='black',alpha=0.1))
ax5.add_patch(Rectangle((20, -35), 2, 60,facecolor='black',alpha=0.1))
ax5.add_patch(Rectangle((30, -35), 2, 60,facecolor='black',alpha=0.1))
ax5.add_patch(Rectangle((40, -35), 2, 60,facecolor='black',alpha=0.1))
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*(BCztw4[idx1:idx2].cpu().detach()),color = 'skyblue',linestyle="-",linewidth=2)
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*(BCnztw4[idx1:idx2].cpu().detach()),color = 'steelblue',linestyle="-",linewidth=2)
plt.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*(BCz2tw4[idx1:idx2].cpu().detach()),color = 'slategray',linestyle=":",linewidth=3)
plt.xlabel('Time [ms]',fontsize=20)
plt.ylabel('$G_{op}$ [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)
plt.xlim(0,50)
plt.ylim(-15,15)
plt.text(0.2, 12, '(B)',fontsize=26)
plt.text(2.2, 12, 'Fully-constrained',fontweight='bold',fontsize=26)

fig.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.1,wspace=0.45,hspace=0.3)

#%% STACK IMAGES

from PIL import Image

image1 = Image.open('FIGURE5a_FINAL.tiff')
image2 = Image.open('FIGURE5b_FINAL.tiff')
image3 = Image.open('FIGURE5c_FINAL.tiff')

width = max(image1.width,image2.width,image3.width)
image1 = image1.resize((width,image1.height))
image2 = image2.resize((width,image2.height))
image3 = image3.resize((width,image3.height))

stacked_image = Image.new('RGB',(width,image1.height+image2.height+image3.height))

stacked_image.paste(image1,(0,0))
stacked_image.paste(image2,(0,image1.height))
stacked_image.paste(image3,(0,image1.height+image2.height))

stacked_image.save('FIGURE5_FINAL.tiff')