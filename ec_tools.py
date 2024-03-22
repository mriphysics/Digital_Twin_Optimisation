# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:17:23 2022

@author: fmglang, dwest
"""

import torch
import math
import util

gamma_ = 42.5764 # MHz/T

def truncate(number, digits) -> float:
    nbDecimals = len(str(number).split('.')[1]) 
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number.item()) / stepper

def EC_perturbation_simple(seq, slew, Ndummies, grad_dir, dummies=(5,20), return_slew=False):
    
    seq_new = seq.clone()
    
    szread = torch.sum(seq_new[Ndummies].adc_usage > 0) + Ndummies
    TR_idx = torch.linspace(0,szread-1,szread).int()      
        
    loop_count = 0
    
    wlength = torch.zeros(szread).to(util.get_device()) 
    
    for gg in TR_idx:
        
        grad_pick = seq_new[gg].gradm[:,grad_dir]
        time_pick = seq_new[gg].event_time
        coarse_tstep = time_pick[1]
        
        waveform = moms2phys(grad_pick[:]) / coarse_tstep
        
        if gg == 0:
            tw_cat = waveform
            wlength[gg] = tw_cat.size(dim=0)
        else:
            tw_cat = torch.cat((tw_cat,waveform), 0) 
            wlength[gg] = tw_cat.size(dim=0) # Cumulative sum.
            
    # Slew rate on raster time.
    sr_rt = (tw_cat[1:] - tw_cat[:-1]) / coarse_tstep 
    sr_rt = torch.cat((sr_rt, torch.zeros([1]).to(util.get_device()))) # Zero pad (end, as anyway gradient is ramped down there).

    # Multi-exponential but just consider one for now.
    maxtime = truncate(tw_cat.shape[0]*coarse_tstep,10) #10dp
    timings = torch.arange(0, maxtime, coarse_tstep).to(util.get_device())

    alphas = [   1,    1, 0] # Paper parameters.
    taus   = [1e-3, 1e-1, 1] # Paper parameters.
    
    ec_perturb = torch.zeros(timings.shape).to(util.get_device())
    for alpha, tau in zip(alphas, taus): # Sum up all exponentials .
        ec_perturb += alpha*torch.exp(-timings/tau)
        
    # Use neural network convolution, this should be hopefully differentiable.
    response = torch.nn.functional.conv1d(sr_rt.reshape([1,1,-1]), # [batch,channels=1,time]
                                          ec_perturb.flip(0).unsqueeze(0).unsqueeze(0), # Flip as conv in machine learning terms is actually crosscorrelation, add singleton for batch & channel.
                                          padding=len(ec_perturb) # Figured out by playing around, not fully sure if always correct.
                                         ).flatten() # Back to original shape.
    # NEEDED FOR NUMERICAL PRECISION ERROR
    if ec_perturb.size(dim=0) > tw_cat.size(dim=0):
        diff = ec_perturb.size(dim=0) - tw_cat.size(dim=0)
        response = response[:len(ec_perturb)-diff]
    elif ec_perturb.size(dim=0) < tw_cat.size(dim=0):
        diff = tw_cat.size(dim=0) - ec_perturb.size(dim=0)
        diff_tensor = torch.zeros(diff)
        response = response[:len(ec_perturb)]
        response = torch.concat([response.cpu(),diff_tensor],-1)
    else:
        response = response[:len(ec_perturb)]
        
    ampl = 1e-5
    tw_perturb = tw_cat - ampl * response # Minus due to Lenz's law.
    
    for gg in TR_idx:

        if gg == 0:
            total_waveform_perturb = tw_perturb[0:wlength[gg].int().item()]
        else:
            total_waveform_perturb = tw_perturb[wlength[gg-1].int().item():wlength[gg].int().item()]

        new_moms = phys2moms(total_waveform_perturb)*coarse_tstep

        grad_moms_mod = seq_new[gg].gradm.clone()
        
        grad_moms_mod[:,grad_dir] = new_moms 

        loop_count = loop_count+1        
        
        seq_new[gg].gradm = grad_moms_mod
        
    if return_slew:
        return seq_new, sr_rt, tw_cat, tw_perturb
    else:
        return seq_new
    
   
def GIRF_perturbation(seq, FOV, GIRF, return_slew=False):
    
    seq_new = seq.clone()
    
    # Extract gradients from different directions.
    gradx = seq_new[0].gradm[:,0].to(torch.float32)
    grady = seq_new[0].gradm[:,1].to(torch.float32)   
    time_pick = seq_new[0].event_time
    gradt = time_pick[1]
    
    # Define gradient waveforms in physcial units.
    twx = moms2phys(gradx[:],FOV) / gradt
    srx = (twx[1:] - twx[:-1]) / gradt 
    srx = torch.cat((srx, torch.zeros([1]).to(util.get_device())))
    twy = moms2phys(grady[:],FOV) / gradt
    sry = (twy[1:] - twy[:-1]) / gradt 
    sry = torch.cat((sry, torch.zeros([1]).to(util.get_device())))
    
    # Define GIRF terms - these are already on the right time axis (10us)
    hxx = torch.from_numpy(GIRF['hxx'])[:,0].to(torch.float32).to(util.get_device())
    hxy = torch.from_numpy(GIRF['hxy'])[:,0].to(torch.float32).to(util.get_device())
    hyx = torch.from_numpy(GIRF['hyx'])[:,0].to(torch.float32).to(util.get_device())
    hyy = torch.from_numpy(GIRF['hyy'])[:,0].to(torch.float32).to(util.get_device())

    # Perform convolutions.
    gxhxx = torch.nn.functional.conv1d(
            twx.unsqueeze(0).unsqueeze(0), hxx.flip(0).unsqueeze(0).unsqueeze(0), padding=len(hxx) // 2
            ).squeeze()
    gyhyy = torch.nn.functional.conv1d(
            twy.unsqueeze(0).unsqueeze(0), hyy.flip(0).unsqueeze(0).unsqueeze(0), padding=len(hyy) // 2
            ).squeeze()
    gxhxy = torch.nn.functional.conv1d(
            twx.unsqueeze(0).unsqueeze(0), hxy.flip(0).unsqueeze(0).unsqueeze(0), padding=len(hxy) // 2
            ).squeeze()
    gyhyx = torch.nn.functional.conv1d(
            twy.unsqueeze(0).unsqueeze(0), hyx.flip(0).unsqueeze(0).unsqueeze(0), padding=len(hyx) // 2
            ).squeeze()
    
    # Deal with inconsistent padding for even and odd sequence lengths.
    if hxx.shape[-1] % 2 == 0:
        gxhxx = gxhxx[:-1]
        gyhyy = gyhyy[:-1]
        gxhxy = gxhxy[:-1]
        gyhyx = gyhyx[:-1]       
    
    # Add terms to get final perturbed gradients.
    gradxp = gxhxx + gyhyx
    gradyp = gyhyy + gxhxy

    # Convert back to moments.
    newmomsx = phys2moms(gradxp,FOV)*gradt   
    newmomsy = phys2moms(gradyp,FOV)*gradt
     
    # Assign these to sequence structure.
    gmomsp = seq_new[0].gradm.clone()
    gmomsp[:,0] = newmomsx  
    gmomsp[:,1] = newmomsy
    seq_new[0].gradm = gmomsp
    
    if return_slew:
        return seq_new, srx, sry, twx, twy, gradxp, gradyp
    else:
        return seq_new

    
def moms2phys(moms, FOV=32e-3): #REMOVE DEFAULT VALUE!
    return moms / FOV / (gamma_*1e6)


def phys2moms(moms, FOV=32e-3): #REMOVE DEFAULT VALUE
    return moms * FOV * (gamma_*1e6)