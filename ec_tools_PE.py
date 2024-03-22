# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:17:23 2022

@author: fmglang
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

def EC_perturbation_simple(seq, slew, Ndummies, grad_dir, dummies=(50,200), return_slew=False):
    
    seq_new = seq.clone()

    szread = 32
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
            wlength[gg] = tw_cat.size(dim=0)
        
    # Slew rate on raster time.
    sr_rt = (tw_cat[1:] - tw_cat[:-1]) / coarse_tstep 
    sr_rt = torch.cat((sr_rt, torch.zeros([1]).to(util.get_device()))) # Zero pad (end, as anyway gradient is ramped down there).
        
    maxtime = truncate(tw_cat.shape[0]*coarse_tstep,10) #10dp
    timings = torch.arange(0, maxtime, coarse_tstep).to(util.get_device())
    alphas = [   1/10,     1/10,    0]
    taus =   [   1e-3,     1e-1,    1]
        
    ec_perturb = torch.zeros(timings.shape).to(util.get_device())
    for alpha, tau in zip(alphas, taus): # Sum up all exponentials. 
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
    
    ampl = 1e-5 # ampl = 0 to check no perturb gives original moments.
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
    
        # Put back to new sequence object.
        seq_new[gg].gradm = grad_moms_mod
    
    if return_slew:
        return seq_new, sr_rt, tw_cat
    else:
        return seq_new

def preemph(seq, slew, gmax, Ndummies, grad_dir, dummies=(50,200), return_slew=False):
    
    seq_new = seq.clone()
    
    szread = len(seq)
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
            wlength[gg] = tw_cat.size(dim=0)
            
    # Slew rate on raster time.
    sr_rt = (tw_cat[1:] - tw_cat[:-1]) / coarse_tstep 
    sr_rt = torch.cat((sr_rt, torch.zeros([1]).to(util.get_device()))) # Zero pad (end, as anyway gradient is ramped down there).

    maxtime = truncate(tw_cat.shape[0]*coarse_tstep,10) #10dp
    timings = torch.arange(0, maxtime, coarse_tstep).to(util.get_device())
    
    # OPTIMIZED VALUES
    alphas = [  0.139169543982,     0.110823690891,    0]
    taus =   [  0.000888323877,     0.090011969209,    1]
    
    ec_perturb = torch.zeros(timings.shape).to(util.get_device())
    for alpha, tau in zip(alphas, taus): # sum up all exponentials 
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
        
    ampl = 1e-5 # ampl = 0 to check no perturb gives original moments.
    tw_perturb = tw_cat + ampl * response # Minus due to Lenz's law.

    for gg in TR_idx:

        if gg == 0:
            total_waveform_perturb = tw_perturb[0:wlength[gg].int().item()]
        else:
            total_waveform_perturb = tw_perturb[wlength[gg-1].int().item():wlength[gg].int().item()]

        new_moms = phys2moms(total_waveform_perturb)*coarse_tstep

        grad_moms_mod = seq_new[gg].gradm.clone()

        grad_moms_mod[:,grad_dir] = new_moms 

        loop_count = loop_count+1        
    
        # Put back to new sequence object.
        seq_new[gg].gradm = grad_moms_mod
        
    if return_slew:
        return seq_new, sr_rt, tw_cat, tw_perturb
    else:
        return seq_new

def EC_perturbation_preemph(seq, slew, Ndummies, grad_dir, dummies=(50,200), return_slew=False):
    
    seq_new = seq.clone()
    
    szread = len(seq)
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
            wlength[gg] = tw_cat.size(dim=0)
            
    # Slew rate on raster time.
    sr_rt = (tw_cat[1:] - tw_cat[:-1]) / coarse_tstep 
    sr_rt = torch.cat((sr_rt, torch.zeros([1]).to(util.get_device()))) # Zero pad (end, as anyway gradient is ramped down there).
    
    # Slew rate on raster time.
    sr_rt = (tw_cat[1:] - tw_cat[:-1]) / coarse_tstep 
    sr_rt = torch.cat((sr_rt, torch.zeros([1]).to(util.get_device()))) # Zero pad (end, as anyway gradient is ramped down there).

    maxtime = truncate(tw_cat.shape[0]*coarse_tstep,10) #10dp
    timings = torch.arange(0, maxtime, coarse_tstep).to(util.get_device())
    alphas = [  1/10,     1/10,    0]
    taus =   [  1e-3,     1e-1,    1]
    
    ec_perturb = torch.zeros(timings.shape).to(util.get_device())
    for alpha, tau in zip(alphas, taus): # Sum up all exponentials. 
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
    
    ampl = 1e-5 # ampl = 0 to check no perturb gives original moments.
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
   
       # Put back to new sequence object.
       seq_new[gg].gradm = grad_moms_mod
        
    if return_slew:
        return seq_new, sr_rt, tw_cat, tw_perturb
    else:
        return seq_new
    
def moms2phys(moms, FOV=32e-3):
    return moms / FOV / (gamma_*1e6)


def phys2moms(moms, FOV=32e-3):
    return moms * FOV * (gamma_*1e6)

def preemph_mod(seq, alpha_in, tau_in, slew, gmax, Ndummies, grad_dir, dummies=(50,200), return_slew=False):
    
    seq_new = seq.clone()
    
    szread = len(seq)
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
            wlength[gg] = tw_cat.size(dim=0)
            
    # Slew rate on raster time.
    sr_rt = (tw_cat[1:] - tw_cat[:-1]) / coarse_tstep 
    sr_rt = torch.cat((sr_rt, torch.zeros([1]).to(util.get_device()))) # Zero pad (end, as anyway gradient is ramped down there).

    maxtime = truncate(tw_cat.shape[0]*coarse_tstep,10) #10dp
    timings = torch.arange(0, maxtime, coarse_tstep).to(util.get_device())
    alphas = alpha_in
    taus =   tau_in
    
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
        
    ampl = 1e-5 # ampl = 0 to check no perturb gives original moments.
    tw_perturb = tw_cat + ampl * response # Minus due to Lenz's law.

    for gg in TR_idx:

        if gg == 0:
            total_waveform_perturb = tw_perturb[0:wlength[gg].int().item()]
        else:
            total_waveform_perturb = tw_perturb[wlength[gg-1].int().item():wlength[gg].int().item()]

        new_moms = phys2moms(total_waveform_perturb)*coarse_tstep

        grad_moms_mod = seq_new[gg].gradm.clone()

        grad_moms_mod[:,grad_dir] = new_moms 

        loop_count = loop_count+1        
    
        # Put back to new sequence object.
        seq_new[gg].gradm = grad_moms_mod
        
    if return_slew:
        return seq_new, sr_rt, tw_cat, tw_perturb
    else:
        return seq_new