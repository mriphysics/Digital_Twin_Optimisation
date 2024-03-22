from __future__ import annotations
import torch
# from .sequence import Sequence
import MRzeroCore as mr0
from typing import Optional
from numpy import pi
import numpy as np
import time
import util
import torchkbnufft as tkbn


def nufft_reco_2D(
    signal: torch.Tensor,
    kspace: torch.Tensor,
    im_size: tuple[int, int],
) -> torch.Tensor:
    """NUFFT reconstruction using torchkbnufft.

    The resolution of the reconstructed image is determined by torchkbnufft
    based on the maximal encoded frequency in the k-space. Changing `im_size`
    will change the FOV of the reconstruction. Multiply the k-space trajectory
    with a constant `x` to increase the FOV by a factor of `x`.

    Parameters
    ----------
    signal : torch.Tensor
        A complex tensor containing the signal, shape (sample_count, 1).
        Multicoil is currently not supported.
    kspace : torch.Tensor
        A real tensor of shape (sample_count, 4) for the kspace trajectory.
    im_size : (int, int)
        Size of the reconstruction, changes the FOV. If too small, only a
        part of the image is seen. If too large, the ROI is only a part of the
        reconstructed image.
    """
    kdata = signal.view(1, 1, -1)
    ktraj = kspace[:, :2].T / (2 * np.pi)

    adjnufft_ob = tkbn.KbNufftAdjoint(im_size, device=util.get_device())
    dcomp = tkbn.calc_density_compensation_function(ktraj, im_size=im_size)
    return adjnufft_ob(kdata * dcomp, ktraj).view(im_size)


def reconstruct(signal: torch.Tensor,
                kspace: torch.Tensor,
                resolution: tuple[int, int, int] | float | None = None,
                FOV: tuple[float, float, float] | float | None = None,
                return_multicoil: bool = False,
                ) -> torch.Tensor:
    """Adjoint reconstruction of the signal, based on a provided kspace.

    Parameters
    ----------
    signal : torch.Tensor
        A complex tensor containing the signal,
        shape (sample_count, coil_count)
    kspace : torch.Tensor
        A real tensor of shape (sample_count, 4) for the kspace trajectory
    resolution : (int, int, int) | float | None
        The resolution of the reconstruction. Can be either provided directly
        as tuple or set to None, in which case the resolution will be derived
        from the k-space (currently only for cartesian trajectories). A single
        float value will be used as factor for a derived resolution.
    FOV : (float, float, float) | float | None
        Because the adjoint reconstruction adapts to the k-space used
        for measurement, scaling gradients will not directly change the FOV of
        the reconstruction. All SimData phantoms have a normalized size of
        (1, 1, 1). Similar to the resolution, a value of None will
        automatically derive the FOV of the sequence based on the kspace. A
        float value can be used to scale this derived FOV.
    return_multicoil : bool
        Specifies if coils should be combined or returned separately.

    Returns
    -------
    torch.Tensor
        A complex tensor with the reconstructed image, the shape is given by
        the resolution.
    """
    res_scale = 1.0
    fov_scale = 1.0
    if isinstance(resolution, float):
        res_scale = resolution
        resolution = None
    if isinstance(FOV, float):
        fov_scale = FOV
        FOV = None

    # Atomatic detection of FOV - NOTE: only works for cartesian k-spaces
    # we assume that there is a sample at 0, 0 nad calculate the FOV
    # based on the distance on the nearest samples in x, y and z direction
    if FOV is None:
        def fov(t: torch.Tensor) -> float:
            t = t[t > 1e-3]
            return 1.0 if t.numel() == 0 else float(t.min())
        tmp = kspace[:, :3].abs()
        fov_x = fov_scale / fov(tmp[:, 0])
        fov_y = fov_scale / fov(tmp[:, 1])
        fov_z = fov_scale / fov(tmp[:, 2])
        FOV = (fov_x, fov_y, fov_z)
        print(f"Detected FOV: {FOV}")

    # Atomatic detection of resolution
    if resolution is None:
        def res(scale: float, fov: float, t: torch.Tensor) -> int:
            tmp = (scale * (fov * (t.max() - t.min()) + 1)).round()
            return max(int(tmp), 1)
        res_x = res(res_scale, FOV[0], kspace[:, 0])
        res_y = res(res_scale, FOV[1], kspace[:, 1])
        res_z = res(res_scale, FOV[2], kspace[:, 2])
        resolution = (res_x, res_y, res_z)
        print(f"Detected resolution: {resolution}")

    # Same grid as defined in SimData
    pos_x = torch.linspace(-0.5, 0.5, resolution[0] + 1)[:-1] * FOV[0]
    pos_y = torch.linspace(-0.5, 0.5, resolution[1] + 1)[:-1] * FOV[1]
    pos_z = torch.linspace(-0.5, 0.5, resolution[2] + 1)[:-1] * FOV[2]
    pos_x, pos_y, pos_z = torch.meshgrid(pos_x, pos_y, pos_z)

    voxel_pos = util.set_device(torch.stack([
        pos_x.flatten(),
        pos_y.flatten(),
        pos_z.flatten()
    ], dim=1)).t()

    NCoils = signal.shape[1]
    # assert NCoils == 1, "reconstruct currently does not support multicoil"

    # (Samples, 4)
    kspace = util.set_device(kspace)
    # (Samples, 3) x (3, Voxels)
    phase = kspace[:, :3] @ voxel_pos
    # (Samples, Voxels): Rotation of all voxels at every event
    rot = torch.exp(2j*pi * phase)  # Matches definition of iDFT

    NCoils = signal.shape[1]

    if return_multicoil:
        return (signal.t() @ rot).view((NCoils, *resolution))
    elif NCoils == 1:
        return (signal.t() @ rot).view(resolution)
    else:
        return torch.sqrt(((torch.abs(signal.t() @ rot))**2).sum(0)).view(resolution)




def get_kmatrix(seq: mr0.Sequence | torch.tensor, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], contrast = 0,
                kspace_scaling: torch.Tensor | torch.Tensor | None = None,
                adc_usage: torch.Tensor | None = None,
                DREAM: bool = False
                ) -> torch.Tensor:
    '''
    reorder scanner signal according to kspace trajectory, works only for
    cartesian (under)sampling (where kspace grid points are hit exactly)
    '''
    # import pdb; pdb.set_trace()
    
    # If seq input is Sequence, generating kspace with function get_kspace()
    # If seq input is the kspace, no further caluclations are necessary
    # Contrast has to be set to 0, no information about adc_usage is available
    
    if not torch.is_tensor(seq):
        kspace = seq.get_kspace()
    else:
        kspace = seq
    NCoils = signal.shape[1]
    
    if kspace_scaling is None:
        kmax = torch.round(torch.max(torch.abs(kspace[:,:3]),0).values)
        kspace_scaling = kmax*2/util.set_device(torch.tensor(resolution))
    
        kspace_scaling[kspace_scaling==0] = 1
    traj = kspace[:,:3]/kspace_scaling
    kindices = (traj + torch.floor(util.set_device(torch.tensor(resolution)) / 2)).round().to(int)
    if contrast and not torch.is_tensor(seq):
        mask = seq.get_contrast_mask(contrast)
        signal = signal[mask]
        kindices = kindices[mask]
        if DREAM:
            kindices[:,0] = kindices[:,0] - torch.min(kindices[:,0])
                
    # import pdb; pdb.set_trace()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(kindices[:,0], kindices[:,1], '.', ms=1)
    # plt.axis('equal')
    
    kmatrix = util.set_device(torch.zeros(*resolution, NCoils, dtype=torch.complex64))
    
    for jj in range(kindices.shape[0]): # I'm sure there must be a way of doing this without any loop...
        ix, iy, iz = kindices[jj,:]
        if ix < 0 or ix >=32 or iy < 0 or iy >=32 or iz < 0 or iz >=1:
            print(f"Invalid indices at iteration {jj}: ix={ix}, iy={iy}, iz={iz}")
            #continue

        kmatrix[ix,iy,iz,:] = signal[jj,:]
        
        
    return kmatrix.permute([3,0,1,2]) # Dim (NCoils x resolution)

def reconstruct_cartesian_fft(seq: mr0.Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], contrast = 0
                ) -> torch.Tensor:
    '''
    do fft reco for Cartesian kspace grid
    '''
    
    ksp = get_kmatrix(seq, signal, resolution, contrast)
    dim = (1,2,3)
    reco_fft = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(ksp,dim=dim),dim=dim),dim=dim)
    # reco_fft = torch.flip(reco_fft, dims=(1,2,3)) # FG: new (i)FT convention
    
    return reco_fft # coils first

def reconstruct_cartesian_fft_naive(seq: mr0.Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], Ndummies, contrast = 0
                ) -> torch.Tensor:
    '''
    do naive fft reco for any kind of signal,
    naive = just assume rectangular kspace matrix (no matter which trajectory was acutally there)
    
    TBD: handle partitions / 3D!
    '''
    
    NRep = len(seq)-Ndummies # NO. DUMMY TRs
    NCol = torch.sum(seq[Ndummies].adc_usage > 0) # assume same number of ADC points in each rep
    
    ksp = signal.reshape([NRep,NCol,1,-1]).permute(3,1,0,2) # coils first,  compensate xy flip
    dim = (1,2,3)
    reco_fft = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(ksp,dim=dim),dim=dim),dim=dim)
    # reco_fft = torch.flip(reco_fft, dims=(1,2,3)) # FG: new (i)FT convention
    
    return reco_fft # coils first

def reconstruct_EPI_fft_naive(seq: mr0.Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], contrast = 0
                ) -> torch.Tensor:
    '''
    do naive fft reco for any kind of signal,
    naive = just assume rectangular kspace matrix (no matter which trajectory was acutally there)
    
    TBD: handle partitions / 3D!
    '''
    
    NRep = resolution[0]
    NCol = resolution[1] 
    
    ksp = signal.reshape([NRep,NCol,1,-1]).permute(3,1,0,2) # coils first,  compensate xy flip
    ksp[:,:,1::2,:] = torch.flip(ksp[:,:,1::2,:], dims=(1,))
    dim = (1,2,3)
    reco_fft = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(ksp,dim=dim),dim=dim),dim=dim)
    # reco_fft = torch.flip(reco_fft, dims=(1,2,3)) # FG: new (i)FT convention
    
    return reco_fft

def reconstruct_cartesian_fft_naive_ZF(seq: mr0.Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], Ndummies, nPF, contrast = 0
                ) -> torch.Tensor:
    '''
    do naive fft reco for any kind of signal,
    naive = just assume rectangular kspace matrix (no matter which trajectory was acutally there)
    
    TBD: handle partitions / 3D!
    '''
    
    NRep = len(seq)-Ndummies # NO. DUMMY TRs
    NCol = torch.sum(seq[Ndummies].adc_usage > 0) # assume same number of ADC points in each rep
    
    ksp = signal.reshape([NRep,NCol,1,-1]).permute(3,1,0,2) # coils first,  compensate xy flip
    tmp = torch.complex(torch.zeros([14,32,32,1]),torch.zeros([14,32,32,1])).to(util.get_device())
    tmp[:,nPF:,:,:] = ksp    
    dim = (1,2,3)
    reco_fft = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(tmp,dim=dim),dim=dim),dim=dim)
    # reco_fft = torch.flip(reco_fft, dims=(1,2,3)) # FG: new (i)FT convention
    
    return reco_fft # coils first

def reconstruct_cartesian_fft_naive_ZF_lowres(seq: mr0.Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], Ndummies, prew_moment, contrast = 0
                ) -> torch.Tensor:
    '''
    do naive fft reco for any kind of signal,
    naive = just assume rectangular kspace matrix (no matter which trajectory was acutally there)
    
    TBD: handle partitions / 3D!
    '''
    
    NRep = len(seq)-Ndummies # NO. DUMMY TRs
    NCol = torch.sum(seq[Ndummies].adc_usage > 0) # assume same number of ADC points in each rep
    
    ksp = signal.reshape([NRep,NCol,1,-1]).permute(3,1,0,2) # coils first,  compensate xy flip
    tmp = torch.zeros([14,10*resolution[0],resolution[1],resolution[2]], dtype=ksp.dtype).to(util.get_device()) # FG: factor 10 and number of coils hard-coded here
    tmp[:,10*resolution[0]-NCol:,(resolution[1]-NRep)//2:(resolution[1]-NRep)//2+NRep,:] = ksp  # put in center (phase-encode direction, less lines), and to edge (read direction, partial-Fourier) 
    dim = (1,2,3)
    reco_fft = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(tmp,dim=dim),dim=dim),dim=dim)
    # reco_fft = torch.flip(reco_fft, dims=(1,2,3)) # FG: new (i)FT convention
    
    return reco_fft # coils first

def remove_oversampling(signal: torch.Tensor, ax=0, oversampling=2):
    # central cropping of signal along axis ax by a given oversampling factor
    sz = signal.shape
    ix = np.array(range(len(sz)))
    signal = signal.permute((ax, *np.setdiff1d(ix, ax))) # put axis that is cropped to front
    lnew = sz[ax]//oversampling # new signal size along cropped axes
    cropix = np.arange(sz[ax]//2 - lnew//2, sz[ax]//2 + lnew//2)
    signal = signal[cropix,:]
    
    rix = np.argsort([ax, *np.setdiff1d(ix, ax)]) # back-permutation to original shape
    return signal.permute((*rix,))

    
def sos(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.abs(x)**2,0))