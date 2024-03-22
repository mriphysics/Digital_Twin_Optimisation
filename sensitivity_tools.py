# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:21:25 2020

@author: fmglang
"""

import numpy as np
import torch
import scipy.io as sio
import cv2


def load_external_coil_sensitivities(path, NCoils, sz):
    loaded = sio.loadmat(path)
    B1minus = loaded['B1minus']
    
    B1minus_rescaled = torch.zeros((NCoils, *sz), dtype=torch.complex64)
    for i in range(NCoils):
        re = cv2.resize(np.real(B1minus[i,:,:]), dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
        im = cv2.resize(np.imag(B1minus[i,:,:]), dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
        
        B1minus_rescaled[i,:,:] = torch.tensor(re) + 1j * torch.tensor(im)
        
    return B1minus_rescaled

def load_external_coil_sensitivities3D(path, NCoils, sz):
    loaded = sio.loadmat(path)
    B1minus = loaded['B1minus']
    
    B1minus_rescaled = torch.zeros((NCoils, *sz), dtype=torch.complex64)
    for i in range(NCoils):
        for j in range(B1minus_rescaled.shape[-1]):
            re = cv2.resize(np.real(B1minus[i,:,:,j]), dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
            im = cv2.resize(np.imag(B1minus[i,:,:,j]), dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
            
            B1minus_rescaled[i,:,:,j] = torch.tensor(re) + 1j * torch.tensor(im)
        
    return B1minus_rescaled

