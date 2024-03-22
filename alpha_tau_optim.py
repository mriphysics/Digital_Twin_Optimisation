# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:50:36 2022

@author: fmglang, dwest
"""

"""3D snapshot GRE sequence."""

import time
import matplotlib.pyplot as plt
import MRzeroCore as mr0
import torch
import os
from seq_builder.GRE3D_EC_builder_PE import GRE3D_EC
import util
from reconstruction import sos, reconstruct_cartesian_fft_naive
import ec_tools_PE
from sensitivity_tools import load_external_coil_sensitivities3D
from termcolor import colored
import numpy as np
import csv

Ndummies_tgt = 0
Ndummies_opt = 0
lobe_dummies1 = (50,200) # Original ratio.

experiment_id = 'alpha_tau'

gmax  = 50e-3 # T/m/s
lobe_dummies2 = (50,200)

##
util.use_gpu = True 
##

smax  = 500   # T/m/s

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
params_target = GRE3D_EC(*size_tmp, Ndummies_tgt, R_accel, dummies = lobe_dummies1)

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

[default_seq, default_sr, default_tw] = ec_tools_PE.EC_perturbation_simple(seq_full, smax, Ndummies_tgt, grad_dir=0, dummies = lobe_dummies1, return_slew=True)

alphas = torch.tensor([0.1,0.1]) # Divided by 10
taus   = torch.tensor([1e-3,1e-1])

global waveform0_x, waveform0p_x, waveform0_y, waveform0p_y
seq1, slew1_x, waveform0_x, waveform0p_x = ec_tools_PE.preemph_mod(seq_full, alphas, taus, smax, gmax, Ndummies_opt, grad_dir=0, dummies = lobe_dummies2, return_slew=True)
seq1, slew1_y, waveform0_y, waveform0p_y = ec_tools_PE.preemph_mod(seq_full, alphas, taus, smax, gmax, Ndummies_opt, grad_dir=1, dummies = lobe_dummies2, return_slew=True)

seq_full_perturbed = ec_tools_PE.EC_perturbation_simple(
                        ec_tools_PE.EC_perturbation_simple(seq_full, smax, Ndummies_tgt, grad_dir=0, dummies = lobe_dummies1),
                        smax, Ndummies_tgt, grad_dir=1, dummies = lobe_dummies1)

# Save gradient moments at position 2.
gmoms2 = torch.zeros(size[0],seq_full[0].gradm.size(0),seq_full[0].gradm.size(1))
for ii in range(size[0]):
    gmoms2[ii] = seq_full_perturbed[Ndummies_tgt+ii].gradm

kloc_perturb = seq_full_perturbed.get_kspace()
kloc_unperturbed = seq_full.get_kspace()

# %% Compare perturbed and unperturbed. 

graph_unperturbed = mr0.compute_graph(seq_full, data, max_state_count, min_state_mag)
graph_perturbed = mr0.compute_graph(seq_full_perturbed, data, max_state_count, min_state_mag)

# Simulate unperturbed.
target_signal_full_unperturbed = mr0.execute_graph(graph_unperturbed, seq_full, target_data)
# target_mag_z_unperturbed = target_signal_full_unperturbed[1]
# target_signal_full_unperturbed = target_signal_full_unperturbed[0]
# target_mag_z_unperturbed = util.to_full(target_mag_z_unperturbed[0][0],data.mask)
# target_mag_z_unperturbed *= util.to_full(data.PD,data.mask)

# Simulate perturbed.
target_signal_full_perturbed = mr0.execute_graph(graph_perturbed, seq_full_perturbed, target_data)
# target_mag_z_perturbed = target_signal_full_perturbed[1]
# target_signal_full_perturbed = target_signal_full_perturbed[0]
# target_mag_z_perturbed = util.to_full(target_mag_z_perturbed[0][0],data.mask)
# target_mag_z_perturbed *= util.to_full(data.PD,data.mask)

# Reconstructions
target_reco_full_unperturbed = reconstruct_cartesian_fft_naive(seq_full,target_signal_full_unperturbed,size,Ndummies_tgt)
target_reco_full_perturbed = reconstruct_cartesian_fft_naive(seq_full_perturbed,target_signal_full_perturbed,size,Ndummies_tgt)

# %% Define loss and helper functions.

torch.cuda.empty_cache()
gif_array = []
loss_history_gauss = []

target = sos(target_reco_full_unperturbed)

class opt_history:
    # For keeping track of stuff during optimization.
    def __init__(self):
        self.loss_history = []
        self.alpha1_history = []
        self.alpha2_history = []
        self.tau1_history = []
        self.tau2_history = []
opt_history = opt_history()

f = open(experiment_id+'.txt','w')

def calc_loss(alphas: torch.Tensor,
              taus: torch.Tensor,
              waveformdx: torch.Tensor,
              waveformdy: torch.Tensor,                           
              params: GRE3D_EC,
              iteration: int):
    
    # MAIN LOSS FUNCTION
    global waveform1_x, waveform1_y, waveform2_x, waveform2_y, waveform3_x, waveform3_y, waveform4_x, waveform4_y, waveform5_x, waveform5_y, waveform6_x, waveform6_y   
    seq = params.generate_sequence()
    seq = mr0.sequence.chain(*seq)
    if util.use_gpu:
        seq = seq.cuda()
    
    # Pre-emphasis to first-order.
    seq, slew1_x, waveform1_x, waveform2_x = ec_tools_PE.preemph_mod(seq, alphas, taus, smax, gmax, Ndummies_opt, grad_dir=0, dummies = lobe_dummies2, return_slew=True)
    seq, slew1_y, waveform1_y, waveform2_y = ec_tools_PE.preemph_mod(seq, alphas, taus, smax, gmax, Ndummies_opt, grad_dir=1, dummies = lobe_dummies2, return_slew=True)

    # EC perturbation keeps old alpha and tau by default.
    seq, slew3_x, waveform3_x, waveform4_x = ec_tools_PE.EC_perturbation_preemph(seq, smax, Ndummies_opt, grad_dir=0, dummies = lobe_dummies2, return_slew=True)
    seq, slew3_y, waveform3_y, waveform4_y = ec_tools_PE.EC_perturbation_preemph(seq, smax, Ndummies_opt, grad_dir=1, dummies = lobe_dummies2, return_slew=True)
    
    global graph  # Just to analyze it in ipython after the script ran.
        
    # Forward simulation.
    signal = mr0.execute_graph(graph, seq, data)
    
    # reco: naive FFT + sum-of-squares coil combine.
    reco = sos(reconstruct_cartesian_fft_naive(seq, signal, size, Ndummies_opt))
    
    # LOSSES
    loss_image = torch.tensor(0.0, device=util.get_device())
    loss_image = util.MSR(reco, target, root=True)
    
    grad1 = torch.cat([waveformdx, waveformdy])
    grad4 = torch.cat([waveform4_x, waveform4_y])
    loss_gamp = torch.tensor(0.0, device=util.get_device())
    loss_gamp = torch.abs(grad1.flatten() - grad4.flatten())
    loss_gamp = torch.sum(loss_gamp)
    
    # Lambdas
    lbd_image = 0
    lbd_gamps = 1e-2
    
    loss = (lbd_image*loss_image +
            lbd_gamps*loss_gamp)
    
    # END LOSSES
    opt_history.loss_history.append(loss.detach().cpu())
    print(f"{alphas[0].detach().cpu():.12f},"+f"{alphas[1].detach().cpu():.12f},"+f"{taus[0].detach().cpu():.12f},"+f"{taus[1].detach().cpu():.12f},"+f"{lbd_gamps*loss_gamp:.12f}\n",file=f)

    print(
        "% 4d |  alpha1 %s | alpha2 %s | tau1 %s | tau2 %s | loss %s | "
        % (
            iteration,
            colored(f"{alphas[0].detach().cpu():.3e}", 'green'),
            colored(f"{alphas[1].detach().cpu():.3e}", 'green'),
            colored(f"{taus[0].detach().cpu():.3e}", 'green'),
            colored(f"{taus[1].detach().cpu():.3e}", 'green'),
            colored(f"{loss.detach().cpu():.3e}", 'yellow'),
        )
    )    
    
    return loss

# %% OPTIMIZATION

### Define the starting parameters for the optimisation process.
size_tmp = [32,32+Ndummies_opt,1]
params = GRE3D_EC(*size_tmp, Ndummies_opt, R_accel, dummies = lobe_dummies2)

params.linearEncoding(adc_count  = params.adc_count,
                      rep_count  = params.rep_count,
                      part_count = params.part_count)

seq_opt = params.generate_sequence()
seq_opt = mr0.sequence.chain(*seq_opt)
if util.use_gpu:
    seq_opt = seq_opt.cuda()

alphas = torch.tensor([0.1,0.1]) # Divided by 10
taus   = torch.tensor([1e-3,1e-1])

alphas.requires_grad = True
taus.requires_grad = True

optimizable_params = [
    {'params': alphas, 'lr': 0.0005},
    {'params': taus,   'lr': 0.0001}
    ]

NRestarts = 1
NIter = 1000

t0 = time.time()

iteration = 0
for restart in range(NRestarts):
    optimizer = torch.optim.Adam(optimizable_params, lr=0.0005, betas = [0.9, 0.999])

    for i in range((restart + 1) * NIter):
        iteration += 1

        if i % 10 == 0:
            graph = mr0.compute_graph(seq_opt, data, max_state_count, min_state_mag)
        
        t1 = time.time()
        print(t1-t0)
        t0 = time.time()
        torch.autograd.set_detect_anomaly(False)
        optimizer.zero_grad()
        loss = calc_loss(alphas, taus, waveform0_x ,waveform0_y, params, iteration)
        loss.backward()
    
        optimizer.step()
        
f.close()

# %% PLOT RESULTS

fileplot = open('alpha_tau.txt') 
csvreader = csv.reader(fileplot)
rows = []
for row in csvreader:
    rows.append(row)
fileplot.close()

del rows[1::2] # Remove odd rows.

row1 = np.zeros(len(rows))
row2 = np.zeros(len(rows))
row3 = np.zeros(len(rows))
row4 = np.zeros(len(rows))

for pp in range(0,len(rows)):
    row1[pp] = rows[pp][0]
    row2[pp] = rows[pp][1]
    row3[pp] = rows[pp][2]
    row4[pp] = rows[pp][3]

plt.figure(1)
plt.subplot(221)
plt.plot(row1,linewidth=2,color='black')
plt.xlabel('Iteration',fontsize=16)
plt.ylabel('alpha1',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.subplot(222)
plt.plot(row2,linewidth=2,color='black')
plt.xlabel('Iteration',fontsize=16)
plt.ylabel('alpha2',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.subplot(223)
plt.plot(row3*1000,linewidth=2,color='black')
plt.xlabel('Iteration',fontsize=16)
plt.ylabel('tau1 [ms]',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.subplot(224)
plt.plot(row4*1000,linewidth=2,color='black')
plt.xlabel('Iteration',fontsize=16)
plt.ylabel('tau2 [ms]',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()