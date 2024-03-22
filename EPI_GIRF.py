#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:50:53 2023

@author: dw16
"""
import sys
import time
import matplotlib.pyplot as plt
from termcolor import colored
import MRzeroCore as mr0
import torch
import os
from seq_builder.GRE3D_EC_builder import EPI2D_EC
import util
from reconstruction import sos, reconstruct_EPI_fft_naive
import ec_tools
from sensitivity_tools import load_external_coil_sensitivities3D
from scipy.io import loadmat
from pulseq_exporter import pulseq_write_EPG_3D_EC_EPI

## SCAN PARAMETERS ##
pulseq         = 0

FOV_export     = 250
FOV_sim        = FOV_export*1e-3

res            = 64

smax           = 180
gmax           = 72e-3

experiment_id = 'EPI_TEST_040324'
path = os.path.dirname(os.path.abspath(__file__))
checkin = None # File name which should be loaded - None

##
util.use_gpu = True 
##

## OPTIMIZATION PARAMETERS ##
MAXITER = 20000
TOTITER = 20000


# %% Loading of simulation data.

# Sequence and reconstruction resolution.
size = (res, res, 1)
size_sim = (res, res, 1)

# load phantom
phantom = mr0.VoxelGridPhantom.brainweb("output/brainweb/subject20.npz")
phantom = phantom.slices([70]).interpolate(*size_sim) # FG: slice 60 seems to match roughly what we had before (70 according to old brainweb data handling)

# load and add Rx coil sensitivities
NCoils = 14
coil_sens = load_external_coil_sensitivities3D('data/B1minus_14ch_simu_3D_Gaussians.mat', NCoils, size_sim)
phantom.coil_sens = coil_sens

if util.use_gpu:
    data = phantom.build().cuda()
else:
    data = phantom.build()
    
data.B0 = torch.tensor(0,dtype=torch.float32).to(util.get_device()) 

# Create target data for mapping.
target_data = data



# pre_pass_settings = (
#     float(torch.mean(data.T1)),
#     float(torch.mean(data.T2)),
#     float(torch.mean(data.T2dash)),
#     float(torch.mean(data.D)),
#     1000, # Number of states (+ and z) simulated in pre-pass.
#     1e-9, # Minimum magnetisation of states in pre-pass.
#     (data.shape / 2).tolist(),
#     data.fov.tolist(),
#     data.avg_B1_trig
# )

max_state_count = 1000
min_state_mag = 1e-9

# %% Simulate target fully sampled.

R_accel = (1,1) # [phase, partition]
size_tmp = [res,res]
params_target = EPI2D_EC(*size_tmp)

seq_full = params_target.generate_sequence()
seq_full = mr0.sequence.chain(*seq_full)
if util.use_gpu:
    seq_full = seq_full.cuda()

GIRF = loadmat('7T_1pcterms.mat')
    
# Export initial sequence: HARD-CODED DEAD-TIME SAMPLES
if pulseq == 1:
    pulseq_write_EPG_3D_EC_EPI(seq_full, 'TEST.seq', FOV_export, GIRF, plot_seq=True, smax=smax)

# %% GIRF PERTURBATION HERE %% #

gmoms1 = seq_full[0].gradm

seq_full_perturbed =  ec_tools.GIRF_perturbation(seq_full, FOV_sim, GIRF)

# Save gradient moments at position 2.
gmoms2 = seq_full_perturbed[0].gradm
    
plt.figure(3)
plt.subplot(231)
plt.plot(gmoms1[:,0].cpu().detach().numpy())
plt.plot(gmoms2[:,0].cpu().detach().numpy())
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.grid()

plt.subplot(232)
plt.plot(gmoms1[:,1].cpu().detach().numpy())
plt.plot(gmoms2[:,1].cpu().detach().numpy())
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.grid()

kloc_perturb = seq_full_perturbed.get_kspace()
kloc_unperturbed = seq_full.get_kspace()

plt.subplot(233)
plt.plot(kloc_unperturbed[:,0].cpu())
plt.plot(kloc_perturb[:,0].cpu())
plt.plot(kloc_unperturbed[:,1].cpu())
plt.plot(kloc_perturb[:,1].cpu())
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.grid()

# %% Compare perturbed and unperturbed. 

graph_unperturbed = mr0.compute_graph(seq_full, data, max_state_count, min_state_mag)
graph_perturbed = mr0.compute_graph(seq_full_perturbed, data, max_state_count, min_state_mag)

# # Simulate unperturbed.
target_signal_full_unperturbed = mr0.execute_graph(graph_unperturbed, seq_full, target_data)
# target_mag_z_unperturbed = target_signal_full_unperturbed[1]
# target_signal_full_unperturbed = target_signal_full_unperturbed[0]
# target_mag_z_unperturbed = util.to_full(target_mag_z_unperturbed[0],data.mask)
# target_mag_z_unperturbed *= util.to_full(data.PD,data.mask)

# Simulate perturbed.
target_signal_full_perturbed = mr0.execute_graph(graph_perturbed, seq_full_perturbed, target_data)
# target_mag_z_perturbed = target_signal_full_perturbed[1]
# target_signal_full_perturbed = target_signal_full_perturbed[0]
# target_mag_z_perturbed = util.to_full(target_mag_z_perturbed[0],data.mask)
# target_mag_z_perturbed *= util.to_full(data.PD,data.mask)

# Reconstructions.
target_reco_full_unperturbed = reconstruct_EPI_fft_naive(seq_full,target_signal_full_unperturbed,size)
target_reco_full_perturbed = reconstruct_EPI_fft_naive(seq_full_perturbed,target_signal_full_perturbed,size)

# %% Define loss and helper functions.

torch.cuda.empty_cache()
gif_array = []
loss_history_gauss = []

target = sos(target_reco_full_unperturbed)
reco_0 = sos(target_reco_full_perturbed)

plt.subplot(234)
plt.imshow(torch.rot90(target.cpu()*1e4))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('viridis')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('x1e-4',fontsize=16)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(14)
plt.xticks(fontsize=14), plt.yticks(fontsize=14),
plt.subplot(235)
plt.imshow(torch.rot90(reco_0.cpu()*1e4))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('viridis')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('x1e-4',fontsize=16)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(14)
plt.xticks(fontsize=14), plt.yticks(fontsize=14),
plt.subplot(236)
plt.imshow(torch.abs(torch.rot90(target.cpu()*1e4)-torch.rot90(reco_0.cpu()*1e4)))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('gray')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('x1e-4',fontsize=16)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(14)
plt.xticks(fontsize=14), plt.yticks(fontsize=14)

class opt_history:
    # For keeping track of stuff during optimization.
    def __init__(self):
        self.loss_history = []
        self.FA = []
opt_history = opt_history()

f = open(experiment_id+'.txt','w')

def calc_loss(gradm_all: torch.Tensor,
              params: EPI2D_EC,
              iteration: int):
    
    # MAIN LOSS FUNCTION
    global gmoms3, gmoms4, xparea, xnarea, yparea, ynarea
    seq = params.generate_sequence()
    seq = mr0.sequence.chain(*seq)
    if util.use_gpu:
        seq = seq.cuda()
    
    # Plug back all grad_moms.
    for jj in range(gradm_all.shape[2]):
        seq[jj].gradm = gradm_all[:,:,jj]
        
    # Save gradient moments at position 3.
    gmoms3 = gradm_all
    
    # EC perturbation.
    seq, slew_x, slew_y, waveform_x,waveform_y, waveformp_x, waveformp_y = ec_tools.GIRF_perturbation(seq, FOV_sim, GIRF, return_slew=True)
    
    global graph  # Just to analyze it in ipython after the script ran.
    
    # Forward simulation.
    signal = mr0.execute_graph(graph, seq, data)
    
    # reco: naive FFT + sum-of-squares coil combine.
    reco = sos(reconstruct_EPI_fft_naive(seq, signal, size))
    
    # Perturbed kspace locations.
    kloc_perturb = seq.get_kspace()
    
    # PLOTTING 
    if (iteration == 1) or (iteration % 10000 == 0):

        plt.figure(figsize=(15, 20))       
        plt.subplot(421)        
        plt.plot(kloc_unperturbed[:,0].cpu().detach().numpy(), kloc_unperturbed[:,1].cpu().detach().numpy(), 'x', label='target')
        plt.plot(kloc_perturb[:,0].cpu().detach().numpy(), kloc_perturb[:,1].cpu().detach().numpy(), '.', label='current')
        plt.title("sampling locations"), plt.legend()                
        plt.subplot(422)
        plt.imshow(util.to_numpy(torch.abs((reco)).transpose(2,1).reshape(size[0],size[1]*size[2])))
        plt.colorbar()
        plt.title("Reco")       
        plt.subplot(424)
        plt.imshow(util.to_numpy((torch.abs(target)).transpose(2,1).reshape(size[0],size[1]*size[2])))
        plt.colorbar()
        plt.title("Target")       
        ax=plt.subplot(423)
        quotients = [number / opt_history.loss_history[0] for number in opt_history.loss_history] # Normalized
        plt.plot(quotients)
        ax.set_yscale('log')
        plt.grid()
        plt.title("Loss Curve")
        
        # Save gradient moments at position 4.
        gmoms4 = torch.cat([rep.gradm.unsqueeze(-1).clone()
                            for rep in seq], dim=2)    

        plt.subplot(413)
        plt.plot(waveform_x.cpu().detach()*1e3, label='waveform x')
        plt.plot(waveform_y.cpu().detach()*1e3, label='waveform y')
        plt.ylabel('waveform (mT/m)'), plt.legend()
        plt.title(f'max = {torch.max(torch.abs(waveform_x).cpu().detach())*1e3:.2f} (x), {torch.max(torch.abs(waveform_y).cpu().detach())*1e3:.2f} (y)')
        
        plt.subplot(414)
        plt.plot(slew_x.cpu().detach(), label='slew x')
        plt.plot(slew_y.cpu().detach(), label='slew y')
        plt.ylabel('slew rate (T/m/s)'), plt.legend()
        plt.title(f'max = {torch.max(torch.abs(slew_x).cpu().detach()):.2f} (x), {torch.max(torch.abs(slew_y).cpu().detach()):.2f} (y)')

        plt.suptitle(f'iter {iteration}')

        gif_array.append(util.current_fig_as_img())
        plt.show()

    # END PLOTTING
    if iteration == 1:
        checkout = {
            'reco_target': target,
            'klocs_target': kloc_unperturbed,
            'klocs_perturbed': kloc_perturb,
            'reco_perturb':reco,
            'slewc':smax,
            'gradc':gmax,          
            }
        torch.save(checkout, os.path.join(path, experiment_id+'1.pth'))
    
    if iteration == TOTITER:
        checkout = {
            'reco_opt': reco,
            'klocs_opt': kloc_perturb,
            'loss_history': opt_history.loss_history,      
            'gmoms1':gmoms1,
            'gmoms2':gmoms2,
            'gmoms3':gmoms3,
            'gmoms4':gmoms4,
            'FAs':params.pulse_angles
            }
        torch.save(checkout, os.path.join(path, experiment_id+'2.pth'))

    # LOSSES
    loss_image = torch.tensor(0.0, device=util.get_device())
    loss_image = util.MSR(reco, target, root=True)
    
    # klocation loss: euclidian distance of kspace sampling locations to 'optimal' ones.
    loss_kloc = torch.sum((torch.abs(kloc_perturb[:,0:3] - kloc_unperturbed[:,0:3])**2).flatten())
    
    # Slew rate penalty.
    slew = torch.cat([slew_x, slew_y])
    loss_slew = torch.tensor(0.0, device=util.get_device())
    loss_slew = torch.abs(slew.flatten()) - smax
    loss_slew[loss_slew < 0] = 0 # Only keep exceeding values.
    loss_slew = torch.sum(loss_slew) # Sum of all slew exceedances.
    
    gamp = torch.cat([waveform_x, waveform_y])
    loss_gamp = torch.tensor(0.0, device=util.get_device())
    loss_gamp = torch.abs(gamp.flatten()) - gmax
    loss_gamp[loss_gamp < 0] = 0 # Only keep exceeding values.
    loss_gamp = torch.sum(loss_gamp) # Sum of all slew exceedances.
    
    indices = torch.tensor(list(range(200)), device=util.get_device()) # NEW RF WAVEFORM LENGTH
    xp_result = torch.index_select(waveformp_x,0,indices)
    yp_result = torch.index_select(waveformp_y,0,indices)
    gampRF = torch.cat([xp_result,yp_result])
    loss_RF = torch.tensor(0.0, device=util.get_device())
    loss_RF = torch.sum(gampRF**2)

    if torch.isnan(loss_image) or torch.isnan(loss_kloc) or torch.isnan(loss_gamp) or torch.isnan(loss_RF):
        print("NaNs PRESENT")
    
    # Lambdas
    lbd_image = 1
    lbd_kloc  = 1
    lbd_slew  = 10000
    lbd_gamps = 10000
    lbd_RF    = 1
    
    loss = (lbd_image*loss_image +
            lbd_kloc*loss_kloc +
            lbd_slew*loss_slew +
            lbd_gamps*loss_gamp + 
            lbd_RF*loss_RF) 
    
    # END LOSSES
    
    opt_history.loss_history.append(loss.detach().cpu())
    
    print(f"{lbd_image*loss_image:.12f},"+f"{lbd_kloc*loss_kloc:.12f},"+f"{lbd_slew*loss_slew:.12f},"+f"{lbd_gamps*loss_gamp:.12f},"+f"{lbd_RF*loss_RF:.12f}\n",file=f) #,"+f"{lbd_area*loss_area:.12f}
    print(
        "% 4d |  image %s | gamp %s | kloc %s | slew %s | RF %s loss %s | "
        % (
            iteration,
            colored(f"{lbd_image*loss_image.detach().cpu():.3e}", 'green'),
            colored(f"{lbd_gamps*loss_gamp.detach().cpu():.3e}", 'green'),
            colored(f"{lbd_kloc*loss_kloc.detach().cpu():.3e}", 'green'),
            colored(f"{lbd_slew*loss_slew.detach().cpu():.3e}", 'green'),
            colored(f"{lbd_RF*loss_RF.detach().cpu():.3e}", 'green'),
            colored(f"{loss.detach().cpu():.3e}", 'yellow'),
        )
    )
    return loss

# %% OPTIMIZATION

### Define the starting parameters for the optimisation process.
size_tmp = [res,res]
params = EPI2D_EC(*size_tmp)

seq_opt = params.generate_sequence()
seq_opt = mr0.sequence.chain(*seq_opt)
if util.use_gpu:
    seq_opt = seq_opt.cuda()

gradm_all = torch.cat([rep.gradm.unsqueeze(-1).clone()
                        for rep in seq_opt], dim=2).to(util.get_device())

gradm_all.requires_grad = True

optimizable_params = [
    {'params': gradm_all, 'lr': 1e-5},
]

NRestarts = 1
NIter = MAXITER

t0 = time.time()

iteration = 0
for restart in range(NRestarts):
    optimizer = torch.optim.Adam(optimizable_params, lr=1e-5, betas = [0.9, 0.999])
        
    for i in range((restart + 1) * NIter):
        iteration += 1

        if i % 10 == 0: # Is re-calculation of graph actually needed if only gradients are optimized?
            graph = mr0.compute_graph(seq_opt, data, max_state_count, min_state_mag)
        
        t1 = time.time()
        print(t1-t0)
        t0 = time.time()
        torch.autograd.set_detect_anomaly(False)
        optimizer.zero_grad()
        
        loss = calc_loss(gradm_all, params, iteration)
        loss.backward()
        optimizer.step()
                
f.close()

# %% Export optimal sequence: HARD-CODED DEAD-TIME SAMPLES

if pulseq == 1:
    seq = params.generate_sequence()
    seq = mr0.sequence.chain(*seq)
    if util.use_gpu:
        seq = seq.cuda()
    
    # Plug back all grad_moms.
    for jj in range(gradm_all.shape[2]):
        seq[jj].gradm = gradm_all[:,:,jj].detach()
        
    pulseq_write_EPG_3D_EC_EPI(seq, 'TEST.seq', FOV_export, GIRF, plot_seq=True, smax=smax+1)