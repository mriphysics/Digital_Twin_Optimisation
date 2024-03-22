#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 08:49:16 2023

@author: dw16
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

gamma_ = 42.5764 # MHz/T
dt  = 10e-6
FOV = 250e-3

def moms2phys(moms, FOV):
    return moms / FOV / (gamma_*1e6)

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray."""
    return x.detach().cpu().numpy()

data1 = torch.load('EPI_TEST_0403241.pth')
data2 = torch.load('EPI_TEST_0403242.pth')

gradc = data1.get('gradc')
slewc = data1.get('slewc')

klocs_perturbed = data1.get('klocs_perturbed')
klocs_target    = data1.get('klocs_target')
klocs_opt       = data2.get('klocs_opt')
reco_target1    = data1.get('reco_target')
reco_perturb  = data1.get('reco_perturb')
reco_target2  = data2.get('reco_target')
reco_opt      = data2.get('reco_opt')

loss_history = data2.get('loss_history')

# Get waveforms from moments.
tw_cat1x = moms2phys(data2.get('gmoms1'),FOV) / dt
tw_cat2x = moms2phys(data2.get('gmoms2'),FOV) / dt
tw_cat3x = moms2phys(data2.get('gmoms3'),FOV) / dt
tw_cat4x = moms2phys(data2.get('gmoms4'),FOV) / dt 

tt = torch.linspace(0,len(tw_cat3x[:,0].cpu().detach().numpy())*dt,len(tw_cat3x[:,0].cpu().detach().numpy())+1)

fig = plt.figure(1)
gs  = fig.add_gridspec(2,6)
ax1 = fig.add_subplot(gs[0,:4])
plt.plot(1000*tt[:-1].cpu().detach(),1000*tw_cat1x[:,0].cpu().detach(),color = 'dimgray',linestyle="--")
plt.plot(1000*tt[:-1].cpu().detach(),1000*tw_cat2x[:,0].cpu().detach(),color = 'lightcoral')
plt.plot(1000*tt[:-1].cpu().detach(),1000*tw_cat3x[:,0].cpu().detach(),color = 'limegreen')
plt.plot(1000*tt[:-1].cpu().detach(),1000*tw_cat4x[:,0].cpu().detach(),color = 'cornflowerblue')
plt.xlabel('Time [ms]',fontsize=20)
plt.ylabel('z Gradient [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)
plt.legend(['$g_{0}$','$G_{0}$','$g_{op}$','$G_{op}$'],fontsize=21,loc='upper right',ncol=4)
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.07, right=0.92,wspace=0.65,hspace=0.1)

klocs_tar = data1.get('klocs_target')
klocs_opt = data2.get('klocs_opt')
klocs_per = data1.get('klocs_perturbed')

ax2 = fig.add_subplot(gs[0,4:])
plt.plot(klocs_per[:,0].cpu().detach().numpy(), klocs_per[:,1].cpu().detach().numpy(), '.',color='lightcoral',markersize=8)
plt.plot(klocs_opt[:,0].cpu().detach().numpy(), klocs_opt[:,1].cpu().detach().numpy(), '.',color='cornflowerblue',markersize=8)
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.xticks([-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20], ['-33','','-31','','-29','','-27','','-25','','-23','','-21',''])
plt.yticks([20,21,22,23,24,25,26,27,28,29,30,31,32,33], ['','21','','23','','25','','27','','29','','31','','33'])
plt.ylim([20,33])
plt.xlim([-33,-20])
ax2.set_ylabel('$k_x$',fontsize=20)
ax2.set_xlabel('$k_z$',fontsize=20)

ax1 = fig.add_subplot(gs[1,:])
plt.plot(1000*tt[:-1].cpu().detach(),1000*((torch.squeeze(tw_cat1x[:,0]).cpu().detach())-((torch.squeeze(tw_cat3x[:,0].cpu().detach())))),color = 'k')
plt.xlabel('Time [ms]',fontsize=20)
plt.ylabel('$g_{0} - g_{op}$ [mT/m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().grid(which='major',axis='x',linewidth=2)
plt.gca().grid(which='minor',axis='x',linewidth=0.5)
plt.gca().grid(which='major',axis='y',linewidth=1)
plt.gca().grid(which='minor',axis='y',linewidth=1)

plt.subplots_adjust(top=0.95, bottom=0.1, left=0.07, right=0.92,wspace=0.7,hspace=0.3)

#%% 

data1 = torch.load('RUN_250mm_64_7ramps1.pth')
data2 = torch.load('RUN_250mm_64_7ramps2.pth')

fig = plt.figure(1)
gs  = fig.add_gridspec(3,7)

reco_t = data1.get('reco_target')
reco_p = data1.get('reco_perturb')
reco_o = data2.get('reco_opt')
max_sig = torch.max(reco_t)

ax4 = fig.add_subplot(gs[0,0])
plot4 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_t/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_xticks([])
ax4.set_yticks([])

ax5 = fig.add_subplot(gs[1,0])
plot5 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((reco_p/max_sig)))))))
plt.set_cmap('gray')
plt.clim(0,1.1)
ax5.set_xticklabels([])
ax5.set_yticklabels([])
ax5.set_xticks([])
ax5.set_yticks([])

ax6 = fig.add_subplot(gs[2,0])
plot6 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_o/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.set_xticks([])
ax6.set_yticks([])

data1 = torch.load('RUN_250mm_64_10ramps1.pth')
data2 = torch.load('RUN_250mm_64_10ramps2.pth')

reco_t = data1.get('reco_target')
reco_p = data1.get('reco_perturb')
reco_o = data2.get('reco_opt')
max_sig = torch.max(reco_t)

ax4 = fig.add_subplot(gs[0,1])
plot4 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_t/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_xticks([])
ax4.set_yticks([])

ax5 = fig.add_subplot(gs[1,1])
plot5 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((reco_p/max_sig)))))))
plt.set_cmap('gray')
plt.clim(0,1.1)
ax5.set_xticklabels([])
ax5.set_yticklabels([])
ax5.set_xticks([])
ax5.set_yticks([])

ax6 = fig.add_subplot(gs[2,1])
plot6 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_o/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.set_xticks([])
ax6.set_yticks([])

data1 = torch.load('RUN_250mm_64_20ramps1.pth')
data2 = torch.load('RUN_250mm_64_20ramps2.pth')

reco_t = data1.get('reco_target')
reco_p = data1.get('reco_perturb')
reco_o = data2.get('reco_opt')
max_sig = torch.max(reco_t)

ax4 = fig.add_subplot(gs[0,2])
plot4 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_t/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_xticks([])
ax4.set_yticks([])

ax5 = fig.add_subplot(gs[1,2])
plot5 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((reco_p/max_sig)))))))
plt.set_cmap('gray')
plt.clim(0,1.1)
ax5.set_xticklabels([])
ax5.set_yticklabels([])
ax5.set_xticks([])
ax5.set_yticks([])

ax6 = fig.add_subplot(gs[2,2])
plot6 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_o/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.set_xticks([])
ax6.set_yticks([])

data1 = torch.load('RUN_250mm_64_30ramps1.pth')
data2 = torch.load('RUN_250mm_64_30ramps2.pth')

reco_t = data1.get('reco_target')
reco_p = data1.get('reco_perturb')
reco_o = data2.get('reco_opt')
max_sig = torch.max(reco_t)

ax4 = fig.add_subplot(gs[0,3])
plot4 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_t/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_xticks([])
ax4.set_yticks([])

ax5 = fig.add_subplot(gs[1,3])
plot5 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((reco_p/max_sig)))))))
plt.set_cmap('gray')
plt.clim(0,1.1)
ax5.set_xticklabels([])
ax5.set_yticklabels([])
ax5.set_xticks([])
ax5.set_yticks([])

ax6 = fig.add_subplot(gs[2,3])
plot6 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_o/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.set_xticks([])
ax6.set_yticks([])

data1 = torch.load('RUN_250mm_64_50ramps1.pth')
data2 = torch.load('RUN_250mm_64_50ramps2.pth')

reco_t = data1.get('reco_target')
reco_p = data1.get('reco_perturb')
reco_o = data2.get('reco_opt')
max_sig = torch.max(reco_t)

ax4 = fig.add_subplot(gs[0,4])
plot4 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_t/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_xticks([])
ax4.set_yticks([])

ax5 = fig.add_subplot(gs[1,4])
plot5 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((reco_p/max_sig)))))))
plt.set_cmap('gray')
plt.clim(0,1.1)
ax5.set_xticklabels([])
ax5.set_yticklabels([])
ax5.set_xticks([])
ax5.set_yticks([])

ax6 = fig.add_subplot(gs[2,4])
plot6 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_o/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.set_xticks([])
ax6.set_yticks([])

data1 = torch.load('RUN_250mm_64_100ramps1.pth')
data2 = torch.load('RUN_250mm_64_100ramps2.pth')

reco_t = data1.get('reco_target')
reco_p = data1.get('reco_perturb')
reco_o = data2.get('reco_opt')
max_sig = torch.max(reco_t)

ax4 = fig.add_subplot(gs[0,5])
plot4 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_t/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_xticks([])
ax4.set_yticks([])

ax5 = fig.add_subplot(gs[1,5])
plot5 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((reco_p/max_sig)))))))
plt.set_cmap('gray')
plt.clim(0,1.1)
ax5.set_xticklabels([])
ax5.set_yticklabels([])
ax5.set_xticks([])
ax5.set_yticks([])

ax6 = fig.add_subplot(gs[2,5])
plot6 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_o/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.set_xticks([])
ax6.set_yticks([])

data1 = torch.load('RUN_250mm_64_100ramps1.pth')
data2 = torch.load('RUN_250mm_64_100ramps2.pth')

reco_t = data1.get('reco_target')
reco_p = data1.get('reco_perturb')
reco_o = data2.get('reco_opt')
max_sig = torch.max(reco_t)

ax4 = fig.add_subplot(gs[0,6])
plot4 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_t/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_xticks([])
ax4.set_yticks([])

ax5 = fig.add_subplot(gs[1,6])
plot5 = plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((reco_p/max_sig)))))))
plt.set_cmap('gray')
plt.clim(0,1.1)
ax5.set_xticklabels([])
ax5.set_yticklabels([])
ax5.set_xticks([])
ax5.set_yticks([])

ax6 = fig.add_subplot(gs[2,6])
plot6 = plt.imshow(np.rot90(to_numpy(torch.abs((reco_o/max_sig)))))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.set_cmap('gray')
plt.clim(0,1.1)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.set_xticks([])
ax6.set_yticks([])

plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.03)