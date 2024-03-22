#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:18:17 2023

@author: dw16
"""

import matplotlib.pyplot as plt
import MRzeroCore as mr0
import torch
import os
from seq_builder.GRE3D_EC_builder import GRE3D_EC
import util
from reconstruction import sos, reconstruct_cartesian_fft_naive,reconstruct_cartesian_fft_naive_ZF, get_kmatrix
import ec_tools
from sensitivity_tools import load_external_coil_sensitivities3D
import numpy as np

PDG_data2 = torch.load('C_BEC_PAPERv32.pth')

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray."""
    return x.detach().cpu().numpy()

FOV_export = 32 # mm

Ndummies_tgt = 300
Ndummies_opt = 0

path = os.path.dirname(os.path.abspath(__file__))
checkin = None

##
util.use_gpu = True 
##

smax    = 500
gmax    = 50e-3

# %% Loading of simulation data.
# Sequence and reconstruction resolution.
size = (32, 32, 1)
size_sim = (32, 32, 1)

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


# Create target data for mapping.
target_data = data


# pre_pass_settings = (
#     float(torch.mean(data.T1)),
#     float(torch.mean(data.T2)),
#     float(torch.mean(data.T2dash)),
#     float(torch.mean(data.D)),
#     1000,
#     1e-9,
#     (data.shape / 2).tolist(),
#     data.fov.tolist(),
#     data.avg_B1_trig
# )

max_state_count = 1000
min_state_mag = 1e-9

# %% Simulate target fully sampled.

R_accel = (1,1) # [phase, partition]
size_tmp = [32,32+Ndummies_tgt,1]
params_target = GRE3D_EC(*size_tmp, Ndummies_tgt, R_accel)

# Different ways of reordering, for now we want 2D linear.
params_target.linearEncoding(adc_count=params_target.adc_count,
                             rep_count=params_target.rep_count,
                             part_count=params_target.part_count)
seq_full = params_target.generate_sequence()
seq_full = mr0.sequence.chain(*seq_full)

if util.use_gpu:
    seq_full = seq_full.cuda()

# %% EDDY CURRENT PERTURBATION HERE %% #

# Save gradient moments at position 1.
gmoms1 = torch.zeros(size[0],seq_full[0].gradm.size(0),seq_full[0].gradm.size(1))
for ii in range(size[0]):
    gmoms1[ii] = seq_full[Ndummies_tgt+ii].gradm

seq_full_perturbed = ec_tools.EC_perturbation_simple(
                        ec_tools.EC_perturbation_simple(seq_full, smax, Ndummies_tgt, grad_dir=0),
                        smax, Ndummies_tgt, grad_dir=1)

kloc_unperturbed = seq_full.get_kspace()

# %% Compare perturbed and unperturbed. 

graph_unperturbed = mr0.compute_graph(seq_full, data, max_state_count, min_state_mag)
graph_perturbed = mr0.compute_graph(seq_full_perturbed, data, max_state_count, min_state_mag)

# Simulate unperturbed.
target_signal_full_unperturbed = mr0.execute_graph(graph_unperturbed, seq_full, target_data,return_mag_z=True)
target_mag_z_unperturbed = target_signal_full_unperturbed[1]
target_signal_full_unperturbed = target_signal_full_unperturbed[0]
kspace_unperturbed = get_kmatrix(seq_full, target_signal_full_unperturbed, size, kspace_scaling=torch.tensor([1.,1.,1.]).to(util.get_device()))
target_mag_z_unperturbed = util.to_full(target_mag_z_unperturbed[0][0],data.mask)
target_mag_z_unperturbed *= util.to_full(data.PD,data.mask)

# Simulate perturbed.
target_signal_full_perturbed = mr0.execute_graph(graph_perturbed, seq_full_perturbed, target_data,return_mag_z=True)
target_mag_z_perturbed = target_signal_full_perturbed[1]
target_signal_full_perturbed = target_signal_full_perturbed[0]
target_mag_z_perturbed = util.to_full(target_mag_z_perturbed[Ndummies_tgt][0],data.mask) # USE 300TH TIMEPOINT

# Reconstructions
target_reco_full_unperturbed = reconstruct_cartesian_fft_naive(seq_full,target_signal_full_unperturbed,size,Ndummies_tgt)
target_reco_full_perturbed   = reconstruct_cartesian_fft_naive(seq_full_perturbed,target_signal_full_perturbed,size,Ndummies_tgt)
target = sos(target_reco_full_unperturbed)

# %% OPTIMIZATION

### Define the starting parameters for the optimisation process.
size_tmp = [32,32+Ndummies_opt,1]
params = GRE3D_EC(*size_tmp, Ndummies_opt, R_accel)

init_mag = torch.abs(target_mag_z_perturbed[data.mask])

# pre_pass_settings2 = (
#     float(torch.mean(init_mag)),
#     float(torch.mean(data.T1)),
#     float(torch.mean(data.T2)),
#     float(torch.mean(data.T2dash)),
#     float(torch.mean(data.D)),
#     1000,
#     1e-9,
#     (data.shape / 2).tolist(),
#     data.fov.tolist(),
#     data.avg_B1_trig
# )

params.linearEncoding(adc_count  = params.adc_count,
                      rep_count  = params.rep_count,
                      part_count = params.part_count)

seq_opt = params.generate_sequence()
seq_opt = mr0.sequence.chain(*seq_opt)

if util.use_gpu:
    seq_opt = seq_opt.cuda()

## GENERATE IMAGE USING GMOMS1
gmoms1 = PDG_data2.get('gmoms1').permute(1,2,0).to(util.get_device())
gmoms_clone = gmoms1.clone()
gmoms = gmoms_clone
for jj in range(gmoms.shape[2]):
    seq_opt[jj].gradm = gmoms[:,:,jj]
    
# No perturbation...
graphUP  = mr0.compute_graph_ss(seq_opt, float(torch.mean(init_mag)), data, max_state_count, min_state_mag) 

signalUP = mr0.execute_graph_ss(graphUP, seq_opt, init_mag, data)
reco_GT  = sos(reconstruct_cartesian_fft_naive_ZF(seq_opt, signalUP, size_tmp, Ndummies_opt, 0))

# Regenerate "optimal" gradient moments.
seq_opt = params.generate_sequence()
seq_opt = mr0.sequence.chain(*seq_opt)

if util.use_gpu:
    seq_opt = seq_opt.cuda()

gmoms3 = PDG_data2.get('gmoms3')
gmoms_clone = gmoms3.clone()
gmoms = gmoms_clone
for jj in range(gmoms.shape[2]):
    seq_opt[jj].gradm = gmoms[:,:,jj]
   
# EC perturbation.
seq_opt, slew_x, waveform_x, waveformp_x = ec_tools.EC_perturbation_simple(seq_opt, smax, Ndummies_opt, grad_dir=0, return_slew=True)
seq_opt, slew_y, waveform_y, waveformp_y = ec_tools.EC_perturbation_simple(seq_opt, smax, Ndummies_opt, grad_dir=1, return_slew=True)

signalP = mr0.execute_graph_ss(graphUP, seq_opt, init_mag, data)
reco_op = sos(reconstruct_cartesian_fft_naive_ZF(seq_opt, signalP, size_tmp, Ndummies_opt, 0))

NRMSE1 = util.NRMSE(reco_op, target)

kloc_1 = seq_opt.get_kspace()

plt.figure(1)
plt.subplot(121)
plt.plot(kloc_unperturbed[:,0].cpu().detach().numpy(), kloc_unperturbed[:,1].cpu().detach().numpy(),'.',markersize=10)
plt.plot(kloc_1[:,0].cpu().detach().numpy(), kloc_1[:,1].cpu().detach().numpy(),'.',markersize=5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.ylim([-20,20])
plt.xlim([-20,20])
plt.ylabel('$k_y$',fontsize=16)
plt.xlabel('$k_x$',fontsize=16)
plt.legend(['GT','Optimized'],fontsize=20,ncol=2)
plt.title('Before Sample Removal',fontsize=24,fontweight='bold')

distance  = torch.norm(kloc_unperturbed-kloc_1,dim=1)
thresh = np.linspace(0,1,101)

NRMSE2 = torch.zeros(101,1)
pcdisc = torch.zeros(101,1)
for jj in range(thresh.shape[0]):
    threshold = thresh[jj]
    ind_filt  = distance >= threshold
    mask_disc = ind_filt.unsqueeze(1).expand_as(kloc_1)
    kloc_2 = kloc_1.clone()
    kloc_2[mask_disc] = 0
    pcdisc[jj] = ((np.count_nonzero(ind_filt.cpu().detach().numpy()))/1024)*100

    # Set data samples that correspond to deviating kloc to zero.
    signalP_masked = signalP.clone()
    signalP_masked[ind_filt,:] = 0
    reco_op_masked = sos(reconstruct_cartesian_fft_naive_ZF(seq_opt, signalP_masked, size_tmp, Ndummies_opt, 0))

    NRMSE2[jj] = util.NRMSE(reco_op_masked, target)

NRMSE_final,idx = torch.min(NRMSE2,dim=0)
threshold_final = thresh[idx.item()]
pcdisc_final = pcdisc[idx.item()]

print(
    "| NRMSE1 %s | NRMSE2 %s | thresh %s | points %s |"
    % (
        f"{NRMSE1}", f"{NRMSE_final.item()}", f"{threshold_final}", f"{pcdisc_final.item()}",
    )
)

threshold = threshold_final
ind_filt  = distance >= threshold
mask_disc = ind_filt.unsqueeze(1).expand_as(kloc_1)
kloc_2 = kloc_1.clone()
kloc_2[mask_disc] = 0
signalP_masked = signalP.clone()
signalP_masked[ind_filt,:] = 0
reco_op_masked = sos(reconstruct_cartesian_fft_naive_ZF(seq_opt, signalP_masked, size_tmp, Ndummies_opt, 0))
#torch.save([kloc_2,reco_op_masked],'C_SEC_PAPERv3_PFresults.pt')