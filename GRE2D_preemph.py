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
from seq_builder.GRE3D_EC_builder_PE import GRE3D_EC, GRE3D_EC_PF
import util

from reconstruction import sos, reconstruct_cartesian_fft_naive, reconstruct_cartesian_fft_naive_ZF_lowres, remove_oversampling

import ec_tools_PE
from sensitivity_tools import load_external_coil_sensitivities3D

Ndummies_tgt = 0
Ndummies_opt = 0
lobe_dummies1 = (50,200) # Original ratio.

alpha1 = 0.139169543982
alpha2 = 0.110823690891
tau1   = 0.000888323877
tau2   = 0.090011969209

alpha_in = torch.tensor([alpha1,alpha2])
tau_in   = torch.tensor([tau1,tau2])  

gmax  = 10e-3 # T/m
lobe_dummies2 = (163,401)

##
util.use_gpu = False 
##

smax  = 500 # T/m/s

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
#     1000,  # Number of states (+ and z) simulated in pre-pass.
#     1e-9,  # Minimum magnetisation of states in pre-pass.
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
# target_mag_z_perturbed = util.to_full(target_mag_z_perturbed[Ndummies_tgt][0],data.mask) # USE 300TH TIMEPOINT

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
        self.FA = []
opt_history = opt_history()

def calc_loss(gradm_all: torch.Tensor,
              params: GRE3D_EC,
              iteration: int):
    
    # MAIN LOSS FUNCTION
    global waveform1_x, waveform1_y, waveform2_x, waveform2_y, waveform3_x, waveform3_y, waveform4_x, waveform4_y, kloc_pe 
    seq = params.generate_sequence()
    seq = mr0.sequence.chain(*seq)
    
    if util.use_gpu:
        seq = seq.cuda()
    
    gradm_all_clone = gradm_all.clone()
    gradm_all = gradm_all_clone
    
    # Plug back all grad_moms.
    for jj in range(gradm_all.shape[2]):
        seq[jj].gradm = gradm_all[:,:,jj]
    
    # Pre-emphasis.
    seq, slew1_x, waveform1_x, waveform2_x = ec_tools_PE.preemph(seq, smax, gmax, Ndummies_opt, grad_dir=0, dummies = lobe_dummies2, return_slew=True)
    seq, slew1_y, waveform1_y, waveform2_y = ec_tools_PE.preemph(seq, smax, gmax, Ndummies_opt, grad_dir=1, dummies = lobe_dummies2, return_slew=True)
    for jj in range(gradm_all.shape[2]):
        gradm_all[:,:,jj] = seq[jj].gradm
    
    kloc_pe = seq.get_kspace()
    
    # EC perturbation.
    seq, slew3_x, waveform3_x, waveform4_x = ec_tools_PE.EC_perturbation_preemph(seq, smax, Ndummies_opt, grad_dir=0, dummies = lobe_dummies2, return_slew=True)
    seq, slew3_y, waveform3_y, waveform4_y = ec_tools_PE.EC_perturbation_preemph(seq, smax, Ndummies_opt, grad_dir=1, dummies = lobe_dummies2, return_slew=True)
    
    global graph  # Just to analyze it in ipython after the script ran.
        
    # Forward simulation.
    signal = mr0.execute_graph(graph, seq, data)
    
    # reco: naive FFT + sum-of-squares coil combine
    reco = sos(reconstruct_cartesian_fft_naive(seq, signal, size, Ndummies_opt))
    
    # Perturbed kspace locations.
    kloc_perturb = seq.get_kspace()
    
    # LOSSES
    loss_image = torch.tensor(0.0, device=util.get_device())
    loss_image = util.MSR(reco, target, root=True)
    
    loss_kboundary = torch.tensor(0.0, device=util.get_device())
    for jj in range(2): # [x,y]
        # Sum up all locations that are outside of boundaries.
        mask_out = (kloc_perturb[:,jj].flatten() > size[jj]/2-1) | (kloc_perturb[:,jj].flatten() < -size[jj]/2)
        kloc_out = kloc_perturb[mask_out,jj]
        loss_kboundary += torch.sum(torch.abs(kloc_out)**2)   
    
    # klocation loss: euclidian distance of kspace sampling locations to 'optimal' ones.
    loss_kloc = torch.sum((torch.abs(kloc_perturb[:,0:3] - kloc_unperturbed[:,0:3])**2).flatten())
    
    # Slew rate penalty.
    slew = torch.cat([slew3_x, slew3_y])
    loss_slew = torch.tensor(0.0, device=util.get_device())
    loss_slew = torch.abs(slew.flatten()) - smax
    loss_slew[loss_slew < 0] = 0 # Only keep exceeding values.
    loss_slew = torch.sum(loss_slew) # Sum of all slew exceedances.
    
    gamp = torch.cat([waveform3_x, waveform3_y])
    loss_gamp = torch.tensor(0.0, device=util.get_device())
    loss_gamp = torch.abs(gamp.flatten()) - gmax
    loss_gamp[loss_gamp < 0] = 0 # Only keep exceeding values.
    loss_gamp = torch.sum(loss_gamp) # Sum of all slew exceedances.
    
    # Lambdas
    lbd_image    = 0
    lbd_boundary = 0
    lbd_kloc     = 20e-6
    lbd_slew     = 1
    lbd_gamps    = 10000
    
    loss = (lbd_image*loss_image +
            lbd_boundary*loss_kboundary +
            lbd_kloc*loss_kloc +
            lbd_slew*loss_slew +
            lbd_gamps*loss_gamp)
    
    # END LOSSES
    
    opt_history.loss_history.append(loss.detach().cpu())
    
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

gradm_all = torch.cat([rep.gradm.unsqueeze(-1).clone()
                        for rep in seq_opt], dim=2).to(util.get_device()) # [NEvent, 3, NRep]

gradm_all.requires_grad = True

optimizable_params = [
    {'params': gradm_all, 'lr': 0.001}]
NRestarts = 1
NIter = 1

t0 = time.time()

iteration = 0
for restart in range(NRestarts):
    optimizer = torch.optim.Adam(optimizable_params, lr=0.001, betas = [0.9, 0.999])

    for i in range((restart + 1) * NIter):
        iteration += 1

        if i % 10 == 0:
            graph = mr0.compute_graph(seq_opt, data, max_state_count, min_state_mag)
        
        t1 = time.time()
        print(t1-t0)
        t0 = time.time()
        torch.autograd.set_detect_anomaly(False)
        optimizer.zero_grad()
        loss = calc_loss(gradm_all, params, iteration)
        loss.backward()
        optimizer.step()

Nsamples  = 32*10 + lobe_dummies2[0] + lobe_dummies2[1] + 23*10 + 20*10
TR_ms     = Nsamples*0.01
TE_ms     = ((32*10)/2 + (20*10)/2 + lobe_dummies2[0])*0.01
origTE_ms = ((32*10)/2 + (20*10)/2 + lobe_dummies1[0])*0.01

coarse_tstep = 1e-5
t_axis = torch.linspace(0,coarse_tstep*(torch.Tensor.size(waveform2_x,0)-1),torch.Tensor.size(waveform2_x,0))

nTR = 32
Gmax_line = torch.ones(nTR*Nsamples)*gmax*1000
plt.figure(3)
plt.plot(t_axis[0:]*1000,waveform2_x[0:].cpu().detach().numpy()*1000,'lightcoral',linewidth=2)
plt.plot(t_axis[0:]*1000,waveform1_x[0:].cpu().detach().numpy()*1000,'cornflowerblue',linewidth=2)
plt.plot(t_axis[0:]*1000,Gmax_line,'k',linewidth=1.5)
plt.plot(t_axis[0:]*1000,-Gmax_line,'k',linewidth=1.5)
plt.grid()
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Time [ms]',fontsize=26,fontweight='bold')
plt.ylabel('Gradient [mT/m]',fontsize=26,fontweight='bold')
plt.ylim(-35,35)
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,hspace=0.2)

min_wave = torch.min(waveform2_x).item()*1000
max_wave = torch.max(waveform2_x).item()*1000

print(
    "| TE %s | TR %s | min %s | max %s |"
    % (
        f"{TE_ms}", f"{TR_ms}", f"{min_wave}", f"{max_wave}",
    )
)

#torch.save([Nsamples,default_tw,waveform2_x,Gmax_line],'PEdata_20mTm.pt')

# %% UNUSED: Start removing samples to get TE back down to 3.1ms.

# no_line = torch.arange(0,17,1) # 16 is maximum because TE only impacts first half of readout.

# TE_new = torch.zeros(17,1)

# for rr in range(len(no_line)):

#     # Remove readout samples (10 per line due to change in dt).
#     ro_points = no_line[rr]*10

#     # Shorten pre-winder (for every line removed, pw does not need to go back as far). 
#     pw_points = no_line[rr]/(torch.abs(min(kloc_unperturbed[:,0]))/lobe_dummies2[0])
    
#     # Calculate new TE.
#     time_remove_ms = ((ro_points + pw_points)*coarse_tstep)*1000
#     TE_new[rr] = TE_ms - time_remove_ms

#     # Continue until TE = TE_orig.
#     if TE_new[rr] <= origTE_ms:
#         break       

# print("DISCARDED LINES = %s" % (f"{no_line[rr].item()}"))

# %% Generate image using sequence with lines removed and compute NRMSE using SS injection!

prew_moment = torch.arange(16,1,-1)

for pp in range(len(prew_moment)):

    # FG: make sure to keep TE the same, thus prewinder has to be stretched when doing partial Fourier along read!
    no_line = 16 - prew_moment[pp].item() # Gets removed from read.
    nprew = lobe_dummies1[0] + 10*no_line # Increase number of prewinder sample to maintain TE when shortening readout  .
    dummies_new = (nprew, lobe_dummies1[1])

    size_tmp = [32,32+Ndummies_opt,1] 
    params = GRE3D_EC_PF(*size_tmp, Ndummies_opt, prew_moment[pp].item(), 1.5, R_accel, dummies = dummies_new) # FG: prewinder has to be stretched to keep TE constant when doing read PF.

    # init_mag = torch.abs(target_mag_z_perturbed[data.mask])
    init_mag = torch.zeros(data.PD.shape) # FG: just disabled for now
    
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
    
    # graph = mr0.compute_graph_ss(seq_opt, *pre_pass_settings2) # FG: does not work currently
    graph = mr0.compute_graph(seq_opt, data, max_state_count, min_state_mag)
    
    seq_opt, slew1_x, waveform1_x, waveform2_x = ec_tools_PE.preemph_mod(seq_opt, alpha_in, tau_in, smax, gmax, Ndummies_opt, grad_dir=0, dummies=dummies_new, return_slew=True)
    seq_opt, slew1_y, waveform1_y, waveform2_y = ec_tools_PE.preemph_mod(seq_opt, alpha_in, tau_in, smax, gmax, Ndummies_opt, grad_dir=1, dummies=dummies_new, return_slew=True)

    seq_opt, slew3_x, waveform3_x, waveform4_x = ec_tools_PE.EC_perturbation_preemph(seq_opt, smax, Ndummies_opt, grad_dir=0, dummies = dummies_new, return_slew=True)
    seq_opt, slew3_y, waveform3_y, waveform4_y = ec_tools_PE.EC_perturbation_preemph(seq_opt, smax, Ndummies_opt, grad_dir=1, dummies = dummies_new, return_slew=True)
    
    # Continue until TE = TE_orig.
    if (torch.min(waveform2_x) > -gmax) & (torch.min(waveform2_y) > -gmax):
        print(f'using prew_moment={prew_moment[pp].item()}')
        break  

spoiler_moment = torch.arange(1.5,0.1,-0.1) # 1.5 is full conventional spoiling, lower (e.g. 1.2, 1.0, 0.8 ...) means less (probably incomplete) spoiling.

for ss in range(len(spoiler_moment)):
    size_tmp = [32,32+Ndummies_opt,1] 
    params = GRE3D_EC_PF(*size_tmp, Ndummies_opt, prew_moment[pp].item(), spoiler_moment[ss].item(), R_accel, dummies = dummies_new)

    # init_mag = torch.abs(target_mag_z_perturbed[data.mask])
    init_mag = torch.zeros(data.PD.shape) # FG: just disabled for now
    
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
    
    # graph = pre_pass.compute_graph_ss(seq_opt, *pre_pass_settings2)
    graph = mr0.compute_graph(seq_opt, data, max_state_count, min_state_mag)
    
    seq_opt, slew1_x, waveform1_x, waveform2_x = ec_tools_PE.preemph_mod(seq_opt, alpha_in, tau_in, smax, gmax, Ndummies_opt, grad_dir=0, dummies=dummies_new, return_slew=True)
    seq_opt, slew1_y, waveform1_y, waveform2_y = ec_tools_PE.preemph_mod(seq_opt, alpha_in, tau_in, smax, gmax, Ndummies_opt, grad_dir=1, dummies=dummies_new, return_slew=True)

    seq_opt, slew3_x, waveform3_x, waveform4_x = ec_tools_PE.EC_perturbation_preemph(seq_opt, smax, Ndummies_opt, grad_dir=0, dummies = dummies_new, return_slew=True)
    seq_opt, slew3_y, waveform3_y, waveform4_y = ec_tools_PE.EC_perturbation_preemph(seq_opt, smax, Ndummies_opt, grad_dir=1, dummies = dummies_new, return_slew=True)
        
    # Continue until TE = TE_orig.
    if (torch.max(waveform2_x) < gmax) & (torch.max(waveform2_y) < gmax):
        break  
    
# signal = execute_graph_ss(graph, seq_opt, init_mag, data) # FG
signal = mr0.execute_graph(graph, seq_opt, data)

reco_new = sos(reconstruct_cartesian_fft_naive_ZF_lowres(seq_opt, signal, size_tmp, Ndummies_opt, prew_moment[pp])) # DEAL WITH 10* IN HERE

# Regenerate target with SS injection.
size_tmp = [32,32+Ndummies_opt,1] 
params = GRE3D_EC_PF(*size_tmp, Ndummies_opt, 16, 1.5, R_accel, dummies = lobe_dummies1)    
params.linearEncoding(adc_count  = params.adc_count,
                      rep_count  = params.rep_count,
                      part_count = params.part_count)   
seq_tgt = params.generate_sequence()
seq_tgt = mr0.sequence.chain(*seq_tgt)
if util.use_gpu:
    seq_tgt = seq_tgt.cuda()

# graph = pre_pass.compute_graph_ss(seq_tgt, *pre_pass_settings2) # FG
# signal_tgt = execute_graph_ss(graph, seq_tgt, init_mag, data)
graph = mr0.compute_graph(seq_tgt, data, max_state_count, min_state_mag) 
signal_tgt = mr0.execute_graph(graph, seq_tgt, data)
target = sos(reconstruct_cartesian_fft_naive(seq_tgt,signal_tgt,size,Ndummies_opt))

NRMSE_PF = util.NRMSE(reco_new, target)

# FG: optional - cut FOV to counter effect of strong oversampling, also do that in NRMSE.
target_noos = remove_oversampling(target, 0, 10)
reco_new_noos = remove_oversampling(reco_new, 0, 10)

NRMSE_PF2 = util.NRMSE(reco_new_noos, target_noos)

plt.figure(1)
plt.subplot(2,3,1)
plt.imshow(target_noos.cpu().detach().numpy()), plt.colorbar(), plt.title('target')
plt.subplot(2,3,2)
plt.imshow(reco_new_noos.cpu().detach().numpy()), plt.colorbar(), plt.title('reco_new')
plt.subplot(2,3,3)
plt.imshow(target_noos.cpu().detach().numpy()- reco_new_noos.cpu().detach().numpy()), plt.colorbar(), plt.title('difference')

kloc = seq_opt.get_kspace()
kloc_orig = seq_full.get_kspace()

plt.figure(1)
plt.subplot(2,3,4)
plt.plot(kloc_orig[:,0].cpu().detach().numpy(), kloc_orig[:,1].cpu().detach().numpy(), '.-')
plt.plot(kloc[:,0].cpu().detach().numpy(), kloc[:,1].cpu().detach().numpy(), '.-')
plt.xlabel('kx'), plt.ylabel('ky'), plt.legend(['PF', 'target'])
plt.subplot(2,3,5)
plt.plot(waveform1_x.cpu().detach().numpy()*1000)
plt.plot(waveform2_x.cpu().detach().numpy()*1000)
plt.subplot(2,3,6)
plt.plot(waveform1_y.cpu().detach().numpy()*1000)
plt.plot(waveform2_y.cpu().detach().numpy()*1000)

disc_points = (1-((32-2*(16-prew_moment[pp]))*(32-(16-prew_moment[pp])))/(32*32))*100

print("NRMSE = %s" % (f"{NRMSE_PF}"))
print("Removed Points (pc) = %s" % (f"{disc_points}"))