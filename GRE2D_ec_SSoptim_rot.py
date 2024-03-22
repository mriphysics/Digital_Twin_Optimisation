# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:50:36 2022

@author: fmglang, dwest
"""

"""3D snapshot GRE sequence."""

import MRzeroCore as mr0
import torch
import time
import matplotlib.pyplot as plt
from termcolor import colored
import os
from seq_builder.GRE3D_EC_builder import GRE3D_EC
import util
from reconstruction import sos, reconstruct_cartesian_fft_naive, get_kmatrix
import ec_tools
from sensitivity_tools import load_external_coil_sensitivities3D
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode

FOV_export = 32 # mm

Ndummies_tgt = 300
Ndummies_opt = 0
experiment_id = 'mLOW_wkDA_v3'
path = os.path.dirname(os.path.abspath(__file__))
checkin = None

##
util.use_gpu = True 
##

MAXITER = 20000
TOTITER = 20000
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

### Rotate here.
def data_augmentation_rot(angle):
    # start with original phantom
    phantom = mr0.VoxelGridPhantom.brainweb("output/brainweb/subject20_3T.npz")
    phantom = phantom.slices([60]).interpolate(*size_sim) # FG: slice 60 seems to match roughly what we had before (70 according to old brainweb data handling)    
    # load and add Rx coil sensitivities
    NCoils = 14
    coil_sens = load_external_coil_sensitivities3D('data/B1minus_14ch_simu_3D_Gaussians.mat', NCoils, size_sim)
    phantom.coil_sens = coil_sens

    def drotate(tensor: torch.Tensor, angle):
        # Weird reshaping to fulfill data shape conventions.
        # full = util.to_full(tensor, data.mask)
        full = tensor.squeeze()
        full = rotate(full.unsqueeze(0), angle, interpolation=InterpolationMode.BILINEAR)
        full = full.squeeze()
        if full.ndim != 3: # to 3D
            full = full.unsqueeze(-1)
        return full

    # modify phantom object with rotated data.
    phantom.PD = drotate(phantom.PD, angle)
    phantom.T1 = drotate(phantom.T1, angle)
    phantom.T2 = drotate(phantom.T2, angle)
    phantom.T2dash = drotate(phantom.T2dash, angle)
    phantom.D = drotate(phantom.D, angle)
    phantom.B0 = drotate(phantom.B0, angle)
    phantom.B1 = drotate(phantom.B1, angle).unsqueeze(0)
    phantom.coil_sens = drotate(phantom.coil_sens, angle).unsqueeze(-1)
    
    if util.use_gpu:
        data_rot = phantom.build().cuda()
    else:
        data_rot = phantom.build()
    
    return data_rot

### Generate roatetd datasets as inputs to the simulation.
NRotDatasets = 20  # Number of augmented datasets with random rotations.
angle_range  = 90  # Angle range for rotation, random uniform between +-angle_range.
TRAINING_DATASETS = [] # Collect all rotated datasets here.
angle=torch.zeros(20)
for jj in range(NRotDatasets):
    angle[jj] = (torch.rand((1,)) - 0.5) * angle_range * 2
    TRAINING_DATASETS.append(data_augmentation_rot(angle[jj].item()))
###

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
target_signal_full_unperturbed = mr0.execute_graph(graph_unperturbed, seq_full, target_data)
# target_mag_z_unperturbed = target_signal_full_unperturbed[1]
# target_signal_full_unperturbed = target_signal_full_unperturbed[0]
kspace_unperturbed = get_kmatrix(seq_full, target_signal_full_unperturbed, size, kspace_scaling=torch.tensor([1.,1.,1.]).to(util.get_device()))
# target_mag_z_unperturbed = util.to_full(target_mag_z_unperturbed[0][0],data.mask)
# target_mag_z_unperturbed *= util.to_full(data.PD,data.mask)

# Simulate perturbed.
target_signal_full_perturbed = mr0.execute_graph(graph_perturbed, seq_full_perturbed, target_data)
# target_mag_z_perturbed = target_signal_full_perturbed[1]
# target_signal_full_perturbed = target_signal_full_perturbed[0]
# target_mag_z_perturbed = util.to_full(target_mag_z_perturbed[Ndummies_tgt][0],data.mask) # USE 300TH TIMEPOINT

# Reconstructions
target_reco_full_unperturbed = reconstruct_cartesian_fft_naive(seq_full,target_signal_full_unperturbed,size,Ndummies_tgt)
target_reco_full_perturbed   = reconstruct_cartesian_fft_naive(seq_full_perturbed,target_signal_full_perturbed,size,Ndummies_tgt)

# %% Simulate database of targets.

TRAINING_TARGETS = [] # Gather all rotated simulated target images.
TRAINING_ZMAG = [] # Gather all rotated simulated z-magnetizations.
for jj in range(NRotDatasets):
    target_signal_rot = mr0.execute_graph(graph_unperturbed, seq_full, TRAINING_DATASETS[jj])
    # target_signal_rot = target_signal_rot[0]
    target_reco_rot = reconstruct_cartesian_fft_naive(seq_full,target_signal_rot,size,Ndummies_tgt)
    TRAINING_TARGETS.append(sos(target_reco_rot))
    
    # Simulate perturbed - this is what we put later as initial_mag.
    target_signal_full_perturbed = mr0.execute_graph(graph_perturbed, seq_full_perturbed, TRAINING_DATASETS[jj])
    # target_mag_z_perturbed = target_signal_full_perturbed[1]
    # target_mag_z_perturbed = util.to_full(target_mag_z_perturbed[Ndummies_tgt][0],TRAINING_DATASETS[jj].mask) # USE 300TH TIMEPOINT
    # TRAINING_ZMAG.append(target_mag_z_perturbed)
    print(f'simulated target {jj+1} of {NRotDatasets}')

# Save simulated targets here.
## FG: DOES NOT WORK : Can't pickle local object 'VoxelGridPhantom.build.<locals>.dephasing_func'
# probably there is a nested function handle in the objects
torch.save([TRAINING_DATASETS, TRAINING_TARGETS, TRAINING_ZMAG], f'training_dataset_nrot{NRotDatasets}_pm{angle_range}deg_v3.pt')

#%% Load training database (from previous computation).

fname = 'training_dataset_nrot20_pm90deg_v3.pt'
TRAINING_DATASETS,  TRAINING_TARGETS, TRAINING_ZMAG = torch.load(fname)

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

data_idx = 0

def calc_loss(gradm_all: torch.Tensor,
              params: GRE3D_EC,
              seq_opt,
              iteration: int):
      
    global data_idx, graph
    # Load them always in the same order, could also shuffle here...
    data   = TRAINING_DATASETS[data_idx % NRotDatasets]
    target = TRAINING_TARGETS[data_idx % NRotDatasets]
    # target_mag_z_perturbed = TRAINING_ZMAG[data_idx % NRotDatasets]
    
    ### Initialize graph with current training sample.
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
    
    # graph = pre_pass.compute_graph_ss(seq_opt, *pre_pass_settings2)
    graph = mr0.compute_graph(seq_opt, data, max_state_count, min_state_mag) # FG: no ss possible currently...
    ###
    
    # Load a new rotated training data set every now and then.
    if iteration % 5 == 0:
        data_idx += 1
    
    # MAIN LOSS FUNCTION
    global gmoms3, gmoms4
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
    seq, slew_x, waveform_x, waveformp_x = ec_tools.EC_perturbation_simple(seq, smax, Ndummies_opt, grad_dir=0, return_slew=True)
    seq, slew_y, waveform_y, waveformp_y = ec_tools.EC_perturbation_simple(seq, smax, Ndummies_opt, grad_dir=1, return_slew=True)
  
    # Forward simulation.
    # signal = execute_graph_ss(graph, seq, init_mag, data) # FG: no ss for first tests, since return of z mag is currently not included
    signal = mr0.execute_graph(graph, seq, data)

    # reco: naive FFT + sum-of-squares coil combine.
    reco = sos(reconstruct_cartesian_fft_naive(seq, signal, size, Ndummies_opt))
    
    # Perturbed kspace locations.
    kloc_perturb = seq.get_kspace()
    
    # PLOTTING 
    if (iteration == 1) or (iteration % 1 == 0):

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
                            for rep in seq], dim=2) # [NEvent,3,NRep]       

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
    
    loss_kboundary = torch.tensor(0.0, device=util.get_device())
    for jj in range(2): # [x,y]
        # Sum up all locations that are outside of boundaries.
        mask_out = (kloc_perturb[:,jj].flatten() > size[jj]/2-1) | (kloc_perturb[:,jj].flatten() < -size[jj]/2)
        kloc_out = kloc_perturb[mask_out,jj]
        loss_kboundary += torch.sum(torch.abs(kloc_out)**2)   
    
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
    
    indices = torch.cat([torch.arange(i,i+20) for i in range(0,3200,100)]).to(util.get_device())
    xp_result = torch.index_select(waveformp_x,0,indices)
    yp_result = torch.index_select(waveformp_y,0,indices)
    gampRF = torch.cat([xp_result,yp_result])
    loss_RF = torch.tensor(0.0, device=util.get_device())
    loss_RF = torch.sum(gampRF**2)
        
    # Lambdas
    lbd_image    = 1
    lbd_boundary = 0
    lbd_kloc     = 10e-8
    lbd_slew     = 1
    lbd_gamps    = 10000
    lbd_RF       = 10000
    
    loss = (lbd_image*loss_image +
            lbd_boundary*loss_kboundary +
            lbd_kloc*loss_kloc +
            lbd_slew*loss_slew +
            lbd_gamps*loss_gamp + 
            lbd_RF*loss_RF)
    
    # END LOSSES
    
    opt_history.loss_history.append(loss.detach().cpu())
    opt_history.FA.append(params.pulse_angles.detach().cpu())

    print(f"{lbd_image*loss_image:.12f},"+f"{lbd_kloc*loss_kloc:.12f},"+f"{lbd_slew*loss_slew:.12f},"+f"{lbd_gamps*loss_gamp:.12f},"+f"{lbd_RF*loss_RF:.12f}\n",file=f)
    print(
        "% 4d |  image %s | gamp %s | kloc %s | slew %s | RF %s | loss %s | "
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
size_tmp = [32,32+Ndummies_opt,1]
params = GRE3D_EC(*size_tmp, Ndummies_opt, R_accel)


params.linearEncoding(adc_count  = params.adc_count,
                      rep_count  = params.rep_count,
                      part_count = params.part_count)

seq_opt = params.generate_sequence()
seq_opt = mr0.sequence.chain(*seq_opt)

if util.use_gpu:
    seq_opt = seq_opt.cuda()

gradm_all = torch.cat([rep.gradm.unsqueeze(-1).clone()
                        for rep in seq_opt], dim=2).to(util.get_device()) # [NEvent, 3, NRep]

gradm_all.requires_grad = True

optimizable_params = [
    {'params': gradm_all, 'lr': 0.001},
]

NRestarts = 1
NIter = MAXITER

t0 = time.time()

iteration = 0
for restart in range(NRestarts):
    optimizer = torch.optim.Adam(optimizable_params, lr=0.001, betas = [0.9, 0.999])
        
    for i in range((restart + 1) * NIter):
        iteration += 1

        t1 = time.time()
        print(t1-t0)
        t0 = time.time()
        torch.autograd.set_detect_anomaly(False)
        optimizer.zero_grad()

        loss = calc_loss(gradm_all, params, seq_opt, iteration)
        loss.backward()
        optimizer.step()
        
f.close()