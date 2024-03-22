#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:46:02 2023

@author: dw16
"""

import torch
import numpy as np
import math
import util

gamma_ = 42.5764 # MHz/T

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray."""
    return x.detach().cpu().numpy()

def moms2phys(moms, FOV=32e-3):
    return moms / FOV / (gamma_*1e6)

def phys2moms(moms, FOV=32e-3):
    return moms * FOV * (gamma_*1e6)

def truncate(number, digits) -> float:
    nbDecimals = len(str(number).split('.')[1]) 
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number.item()) / stepper

data2 = torch.load('UC_test2.pth') # CHANGE FILENAME

gmoms = data2.get('gmoms3')[:,0,:]
#gmoms = data2.get('gmoms1')[:,:,0]

szread = 32
TR_idx = torch.linspace(0,szread-1,szread).int()      
    
wlength = torch.zeros(szread)

for gg in TR_idx:
    
    grad_pick = gmoms[:,gg]
    #grad_pick = gmoms[gg,:]
    coarse_tstep = 100e-6
    
    waveform = moms2phys(grad_pick[:]) / coarse_tstep
    
    if gg == 0:
        tw_cat = waveform
        wlength[gg] = tw_cat.size(dim=0)
    else:
        tw_cat = torch.cat((tw_cat,waveform), 0) 
        wlength[gg] = tw_cat.size(dim=0)
        
sr_rt = (tw_cat[1:] - tw_cat[:-1]) / coarse_tstep 
sr_rt = torch.cat((sr_rt, torch.zeros([1]).to(util.get_device()))) #.to(util.get_device())))

maxtime = truncate(tw_cat.shape[0]*coarse_tstep,10) 
timings = torch.arange(0, maxtime, coarse_tstep)

alphas = [   1,    1, 0]
taus   = [1e-3, 1e-1, 1]

ec_perturb = torch.zeros(timings.shape)
for alpha, tau in zip(alphas, taus):
    ec_perturb += alpha*torch.exp(-timings/tau)

# .to(util.get_device())
response = torch.nn.functional.conv1d(sr_rt.reshape([1,1,-1]),ec_perturb.flip(0).to(util.get_device()).unsqueeze(0).unsqueeze(0),padding=len(ec_perturb)).flatten()

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
tw_perturb = tw_cat - ampl * response

# SAVE RESULTS
torch.save([tw_cat,tw_perturb],'UC_test_TWs.pt')