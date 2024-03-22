# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:50:36 2022

@author: fmglang
"""

"""3D snapshot GRE sequence."""

import matplotlib.pyplot as plt
import MRzeroCore as mr0
import torch
import os
from seq_builder.GRE3D_EC_builder import GRE3D_EC
import util
from reconstruction import sos, reconstruct_cartesian_fft_naive, get_kmatrix
import ec_tools
from sensitivity_tools import load_external_coil_sensitivities3D
import numpy as np

PDG_data2 = torch.load('UC_BEC_PAPERv32.pth')

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray."""
    return x.detach().cpu().numpy()

FOV_export = 32 # mm

Ndummies_tgt = 300
Ndummies_opt = 0
experiment_id = 'TEST'
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

# %% Define loss and helper functions.

torch.cuda.empty_cache()
gif_array = []
loss_history_gauss = []

target = sos(target_reco_full_unperturbed)

plt.imshow(torch.rot90(target.cpu()*1e4))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('x1e-4',fontsize=16)
plt.set_cmap('viridis')
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(14)

class opt_history:
    # For keeping track of stuff during optimization.
    def __init__(self):
        self.loss_history = []
        self.FA = []
opt_history = opt_history()

f = open(experiment_id+'.txt','w')

# %% OPTIMIZATION

### Define the starting parameters for the optimisation process.
size_tmp = [32,32+Ndummies_opt,1]
params = GRE3D_EC(*size_tmp, Ndummies_opt, R_accel) 

mag0 = torch.abs(target_mag_z_perturbed[data.mask])

# pre_pass_settings2 = (
#     float(torch.mean(mag0)),
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
                      rep_cwount  = params.rep_count,
                      part_count = params.part_count)

seq_opt = params.generate_sequence()
seq_opt = mr0.sequence.chain(*seq_opt)

if util.use_gpu:
    seq_opt = seq_opt.cuda()

# UNPERTURBED
gmoms3 = PDG_data2.get('gmoms1').to(util.get_device())
gmoms_clone = gmoms3.clone()
gmoms = gmoms_clone
for ii in range(size_tmp[0]):
    seq_opt[ii].gradm = gmoms[ii,:,:] 
    
graphU = mr0.compute_graph_ss(seq_opt, float(torch.mean(mag0)), data, max_state_count, min_state_mag)
signalU = mr0.execute_graph_ss(graphU, seq_opt, mag0, data)
reco_GT = sos(reconstruct_cartesian_fft_naive(seq_opt, signalU, size_tmp, Ndummies_opt))

# PERTURBED
size_tmp = [32,32+Ndummies_opt,1]
params = GRE3D_EC(*size_tmp, Ndummies_opt, R_accel) 
params.linearEncoding(adc_count  = params.adc_count,
                      rep_count  = params.rep_count,
                      part_count = params.part_count)
seq_opt = params.generate_sequence()
seq_opt = mr0.sequence.chain(*seq_opt)
gmoms3 = PDG_data2.get('gmoms3')
gmoms_clone = gmoms3.clone()
gmoms = gmoms_clone

for ii in range(gmoms.shape[2]):
    seq_opt[ii].gradm = gmoms[:,:,ii]

seq_opt, slew_x, waveform_x, waveformp_x = ec_tools.EC_perturbation_simple(seq_opt, smax, Ndummies_opt, grad_dir=0, return_slew=True)
seq_opt, slew_y, waveform_y, waveformp_y = ec_tools.EC_perturbation_simple(seq_opt, smax, Ndummies_opt, grad_dir=1, return_slew=True)

[td,tt,tz] = torch.load('training_dataset_nrot20_pm90deg_v3.pt')

td.append(data)
tt.append(target)
tz.append(target_mag_z_perturbed)

diff_DA = np.zeros((21))
                   
for zz in range(21):

    data   = td[zz]  
    zmag   = tz[zz][data.mask] 
    target = tt[zz] # Error versus true steady-state sim.
    
    signalP = mr0.execute_graph_ss(graphU, seq_opt, zmag, data)

    reco = sos(reconstruct_cartesian_fft_naive(seq_opt, signalP, size_tmp, Ndummies_opt))
    
    diff_DA[zz] = util.NRMSE(reco, target)

torch.save(diff_DA,'DA_errorsHIGH_v3.pt')