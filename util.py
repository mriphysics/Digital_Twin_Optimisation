"""This module contains helper functions only."""

from __future__ import annotations
from typing import Tuple
import os
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
# import pre_pass
from torch.nn.functional import interpolate
# from . import sequence
# from .sim_data.raw_sim_data import RawSimData
from skimage.metrics import structural_similarity as ssim

use_gpu = True
gpu_dev = 0

# def simple_compute_graph(
#     seq: sequence.Sequence, data: RawSimData,
#     max_state_count: int = 200, min_state_mag: float = 1e-4
# ):
#     """Like pre_pass.compute_graph, but computes args from `` data``."""
#     return pre_pass.compute_graph(
#         seq,
#         float(torch.mean(data.T1)),
#         float(torch.mean(data.T2)),
#         float(torch.mean(data.T2dash)),
#         float(torch.mean(data.D)),
#         max_state_count,
#         min_state_mag,
#         data.nyquist,
#         data.fov.tolist(),
#         data.avg_B1_trig
#     )


def get_device() -> torch.device:
    """Return the device as given by ``util.use_gpu`` and ``util.gpu_dev``."""
    if use_gpu:
        return torch.device(f"cuda:{gpu_dev}")
    else:
        return torch.device("cpu")


def set_device(x: torch.Tensor) -> torch.Tensor:
    """Set the device of the passed tensor as given by :func:`get_deivce`."""
    if use_gpu:
        return x.cuda(gpu_dev)
    else:
        return x.cpu()


def phase_cycler(pulse: int, dphi: float = 137.50776405) -> float:
    """Generate a phase for cycling through phases in a sequence.

    The default value of 360° / Golden Ratio seems to work well, better than
    angles like 117° which produces very similar phases for every 3rd value.

    Parameters
    ----------
        pulse : int
            pulse number for which the phase is calculated
        dphi : float
            phase step size in degrees

    Returns
    -------
        Phase of the given pulse
    """
    return float(np.fmod(0.5 * dphi * (pulse**2+pulse+2), 360) * np.pi / 180)


def current_fig_as_img(dpi: float = 180) -> np.ndarray:
    """Return the current matplotlib figure as image.

    Parameters
    ----------
    dpi : float
        The resolution of the returned image

    Returns
    -------
    np.ndarray
        The current matplotlib figure converted to a 8 bit rgb image.
    """
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", dpi=dpi)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


def to_full(sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Convert a sparse to a full tensor by filling indices given by mask.

    Parameters
    ----------
    sparse : torch.Tensor)
        Sparse tensor containing the data.
    mask : torch.Tensor)
        Mask indicating the indices of the elements in ``sparse``

    Raises
    ------
    ValueError
        If ``mask`` requires more or less elements than ``sparse`` contains.

    Returns
    -------
    torch.Tensor
        The full tensor that has the same shape as ``mask`` and contains the
        data of ``sparse``.
    """
    if mask.count_nonzero() != sparse.shape[-1]:
        raise ValueError(
            f"mask requires {mask.count_nonzero()} elements, "
            f"but sparse contains {sparse.shape[-1]}."
        )
    # coil_sens and B1 have an additional dimension for coils
    if sparse.squeeze().dim() > 1:
        full = torch.zeros(sparse.shape[:-1] + mask.shape,
                           dtype=sparse.dtype, device=sparse.device)
        full[..., mask] = sparse
    else:
        full = torch.zeros(mask.shape,
                           dtype=sparse.dtype, device=sparse.device)
        full[mask] = sparse
    return full


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray."""
    return x.detach().cpu().numpy()


def to_torch(x: np.ndarray) -> torch.Tensor:
    """Convert a numpy ndarray to a torch tensor."""
    return torch.tensor(x, dtype=torch.float)

def plot3D(x: torch.Tensor,figsize=(16,8)) -> None:
    """Plot absolute image of a 3D tensor (x,y,z)
    or 4D tensor (coil,x,y,z)."""
    if x.ndim == 4:
        x = torch.sum(torch.abs(x),0)
    plt.figure(figsize=figsize)
    if type(x).__module__ == np.__name__:
        plt.imshow(np.flip(x,1).transpose(1,2,0).reshape(x.shape[1],x.shape[0]*x.shape[2]))
    else:
        plt.imshow(np.flip(to_numpy(x),1).transpose(1,2,0).reshape(x.shape[1],x.shape[0]*x.shape[2]))
    plt.colorbar()

def complex_loss(input, target):
    eps = 1e-10
    real_input = input[...,0]
    imag_input = input[...,1]
    real_target = target[...,0]
    imag_target = target[...,1]        
    mag_input = torch.sqrt(real_input**2+imag_input**2+eps)
    mag_target = torch.sqrt(real_target**2+imag_target**2+eps)
    angle_loss = torch.mean(torch.abs(real_input*imag_target-imag_input*real_target)/(mag_target+eps))
    angle_loss[torch.isnan(angle_loss)] = 0
    mag_loss = torch.nn.L1Loss()(mag_input,mag_target)
    return mag_loss + angle_loss


def SSIM(a: torch.Tensor, b: torch.Tensor,
         window_size: float = 4.0) -> torch.Tensor:
    """Calculate the structural similarity of two 2D tensors.

    Structural similarity is a metric that tries to estimate how similar two
    images look for humans. The calculated value is per-pixel and describes how
    different or similar that particular pixel looks. While doing so it takes
    the neighbourhood into account, as given by the ``window_size``.

    Parameters
    ----------
    a : torch.Tensor
        A 2D, real valued tensor
    b : torch.Tensor
        A tensor with identical properties as ``a``
    window_size : float
        The window size used when comparing ``a`` and ``b``

    Returns
    -------
    torch.Tensor
        A tensor with the same shape as ``a`` and ``b``, containing for every
        pixel a value between 0 (no similarity) to 1 (identical).
    """
    assert a.shape == b.shape and a.dim() == 2

    x, y = torch.meshgrid([torch.arange(a.shape[0]), torch.arange(a.shape[1])])
    norm = 1 / (2*np.pi*np.sqrt(window_size))

    def gauss(x0: float, y0: float):
        return norm * torch.exp(-((x-x0)**2 + (y-y0)**2) / (2*window_size))

    ssim = torch.zeros_like(a)
    c1 = 1e-4
    c2 = 9e-4

    for x0 in range(a.shape[0]):
        for y0 in range(a.shape[1]):
            window = gauss(x0, y0)
            a_w = a * window
            b_w = b * window

            a_mean = a_w.mean()
            b_mean = b_w.mean()
            a_diff = a_w - a_mean
            b_diff = b_w - b_mean

            ssim[x0, y0] = (
                (
                    (2*a_mean*b_mean + c1)
                    * (2*(a_diff*b_diff).mean() + c2)
                ) / (
                    (a_mean**2 + b_mean**2 + c1)
                    * ((a_diff**2).mean() + (b_diff**2).mean() + c2)
                )
            )

    return ssim


def load_optimizer(optimizer: torch.optim.Optimizer,
                   path: torch.Tensor,
                   NN: torch.nn.Module | None = None
                   ) -> tuple[torch.optim.Optimizer, torch.Tensor,
                              torch.Tensor, torch.Tensor,
                              torch.nn.Module | None]:
    """Load state of optimizer for retraining/restarts

    Parameters
    ----------
    optimizer : torch.optim
        A optimizer
    path : torch.Tensor
        A tensor with the path to the file which sould be loaded

    Returns
    -------
    optimizer : torch.optim
        Optimizer with loaded parameters.
    loss_history : torch.Tensor
        Old loss_history.
    params_target : torch.Tensor
        Sequence parameters for target.
    target_reco : torch.Tensor
        Target reconstruction
    """
    checkin = torch.load(path)
    optimizer.load_state_dict(checkin['optimizer'])
    optimizer.param_groups = checkin['optimizer_params']
    if NN:
        NN.load_state_dict(checkin['NN'])

    return (
        optimizer,
        checkin['loss_history'],
        checkin['params_target'],
        checkin['target_reco'],
        NN
    )


def L1(a: torch.Tensor, b: torch.Tensor,
       absolut: bool = False) -> torch.Tensor:
    """Calculate the L1 norm of two 2D tensors.

    Parameters
    ----------
    a : torch.Tensor
        A 2D, real or imaginar valued tensor
    b : torch.Tensor
        A tensor with identical properties as ``a``
    absolut : bool
        The flag ``absolut`` indicates if the abs() of ``a`` and ``b`` size is
        taken before calculating the L1 norm.

    Returns
    -------
    torch.Tensor
        A tensor with the L1 norm.
    """
    assert a.shape == b.shape

    if absolut:
        norm = torch.sum(torch.abs(torch.abs(a)-torch.abs(b)))
    else:
        norm = torch.sum(torch.abs(a-b))
    return norm


def MSR(a: torch.Tensor, b: torch.Tensor,
        root: bool = False, weighting: torch.Tensor | float = 1,
        norm: bool = False) -> torch.Tensor:
    """Calculate the (R)MSR norm of two 2D tensors.

    Parameters
    ----------
    a : torch.Tensor
        A 2D, real or imaginar valued tensor
    b : torch.Tensor
        A tensor with identical properties as ``a``
    root : torch.bool
        The flag ``root indicates if the square root of the RMS is used.
    weighting : torch.Tensor
        Give a weighting on a and b
    norm : torch.bool
        Gives the normalized MSR on b

    Returns
    -------
    torch.Tensor
        A tensor with the (R)MSE norm.
    """
    assert a.shape == b.shape

    tmp = torch.abs(a*weighting - b*weighting)
    tmp = tmp**2

    tmp = torch.sum(tmp)
    if root:
        tmp = torch.sqrt(tmp)
        #tmp = tmp
        
    if norm:
        tmp /= torch.sum(torch.abs(b*weighting))
        #tmp = tmp    
        
    return tmp

def NRMSE(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    diff = a - b
    squared_diff = diff ** 2

    # Calculate RMSE
    rmse = torch.sqrt(torch.mean(squared_diff))

    # Calculate the range of values in tensor a
    denom = torch.sqrt(torch.mean(b**2))

    # Calculate NRMSE
    nrmse = (rmse / denom) * 100
    
    return nrmse

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    max_pixel = 1.0  # assuming pixel values are normalized between 0 and 1
    psnr_val = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr_val.item()

def calculate_ssim(img1, img2):
    # Convert PyTorch tensors to numpy arrays
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    # Calculate SSIM
    ssim_val, _ = ssim(img1_np[:,:,0], img2_np[:,:,0], multichannel=True, full=True, data_range = np.max([img1_np,img2_np])-np.min([img1_np,img2_np]))
    return ssim_val

def plot_kspace_trajectory(seq: sequence.Sequence,
                           figsize: tuple[float, float] = (5, 5),
                           plotting_dims: str = 'xy',
                           plot_timeline: bool = True,
                           new_figure: bool = True) -> None:
    """Plot the kspace trajectory produced by self.

    Parameters
    ----------
    kspace : list[Tensor]
        The kspace as produced by ``Sequence.get_full_kspace()``
    figsize : (float, float), optional
        The size of the plotted matplotlib figure.
    plotting_dims : string, optional
        String defining what is plotted on the x and y axis ('xy' 'zy' ...)
    plot_timeline : bool, optional
        Plot a second subfigure with the gradient components per-event.
    """
    assert len(plotting_dims) == 2
    assert plotting_dims[0] in ['x', 'y', 'z']
    assert plotting_dims[1] in ['x', 'y', 'z']
    dim_map = {'x': 0, 'y': 1, 'z': 2}

    # TODO: We could (optionally) plot which contrast a sample belongs to,
    # currently we only plot if it is measured or not

    kspace = seq.get_full_kspace()
    adc_mask = [rep.adc_usage > 0 for rep in seq]

    cmap = plt.get_cmap('rainbow')
    if new_figure:
        plt.figure(figsize=figsize)
    if plot_timeline:
        plt.subplot(211)
    for i, (rep_traj, mask) in enumerate(zip(kspace, adc_mask)):
        kx = to_numpy(rep_traj[:, dim_map[plotting_dims[0]]])
        ky = to_numpy(rep_traj[:, dim_map[plotting_dims[1]]])
        measured = to_numpy(mask)

        plt.plot(kx, ky, c=cmap(i / len(kspace)))
        plt.plot(kx[measured], ky[measured], 'r.')
        plt.plot(kx[~measured], ky[~measured], 'k.')
    plt.xlabel(f"$k_{plotting_dims[0]}$")
    plt.ylabel(f"$k_{plotting_dims[1]}$")
    plt.grid()

    if plot_timeline:
        plt.subplot(212)
        event = 0
        for i, rep_traj in enumerate(kspace):
            x = np.arange(event, event + rep_traj.shape[0], 1)
            event += rep_traj.shape[0]
            rep_traj = to_numpy(rep_traj)

            if i == 0:
                plt.plot(x, rep_traj[:, 0], c='r', label="$k_x$")
                plt.plot(x, rep_traj[:, 1], c='g', label="$k_y$")
                plt.plot(x, rep_traj[:, 2], c='b', label="$k_z$")
            else:
                plt.plot(x, rep_traj[:, 0], c='r', label="_")
                plt.plot(x, rep_traj[:, 1], c='g', label="_")
                plt.plot(x, rep_traj[:, 2], c='b', label="_")
        plt.xlabel("Event")
        plt.ylabel("Gradient Moment")
        plt.legend()
        plt.grid()

    if new_figure:
        plt.show()


# TODO: This is specific to GRE-like sequences, make it more general!
def get_signal_from_real_system(path, seq, NRep: float | None = None):
    if NRep is None:
        NRep = len(seq)
    NCol = torch.count_nonzero(seq[2].adc_usage).item()

    print('waiting for TWIX file from the scanner... ' + path)
    done_flag = False
    while not done_flag:    
        if os.path.isfile(path):
            # read twix file
            print("TWIX file arrived. Reading....")

            ncoils = 20
            time.sleep(0.2)
            raw = np.loadtxt(path)

            heuristic_shift = 4
            print("raw size: {} ".format(raw.size) + "expected size: {} ".format("raw size: {} ".format(NRep*ncoils*(NCol+heuristic_shift)*2)) )

            if raw.size != NRep*ncoils*(NCol+heuristic_shift)*2:
                  print("get_signal_from_real_system: SERIOUS ERROR, TWIX dimensions corrupt, returning zero array..")
                  raw = np.zeros((NRep,ncoils,NCol+heuristic_shift,2))
                  raw = raw[:,:,:NCol,0] + 1j*raw[:,:,:NCol,1]
            else:
                  raw = raw.reshape([NRep,ncoils,NCol+heuristic_shift,2])
                  raw = raw[:,:,:NCol,0] + 1j*raw[:,:,:NCol,1]

            # raw = raw.transpose([1,2,0]) #ncoils,NRep,NCol
            raw = raw.transpose([0,2,1]) #NRep,NCol,NCoils
            raw = raw.reshape([NRep*NCol,ncoils])
            raw = np.copy(raw)
            done_flag = True

    return torch.tensor(raw,dtype=torch.complex64)


def write_data_to_seq_file(seq: sequence.Sequence, file_name: str):
    """Write all sequence data needed for reconstruction into a .seq file.

    The data is compressed, base64 encoded and inserted as a comment into the
    pulseq .seq file, which means it is ignored by all interpreters and only
    slightly increases the file size.

    Parameters
    ----------
    seq : Sequence
        Should be the sequence that was used to produce the .seq file
    file_name : str
        The file name to append the data to, it is not checked if this
        actually is a pulseq .seq file.
    """
    kspace = seq.get_kspace().detach()
    adc_usage = torch.cat([rep.adc_usage[rep.adc_usage > 0] for rep in seq])

    # Transpose for more efficient compression (contiguous components)
    kspace_enc = np.ascontiguousarray(kspace.T.cpu().numpy())
    # Delta encoding (works very well for cartesian trajectories)
    kspace_enc[:, 1:] -= kspace_enc[:, :-1]
    # Reduce precision, don't need 32bit for a kspace
    kspace_enc = kspace_enc.astype(np.float16)

    # Compressing adc_usage
    assert -128 <= adc_usage.min() <= 127, "8 bit are not enough"
    adc_usage_enc = adc_usage.cpu().numpy().astype(np.int8)

    # Compress and encode with base64 to write as legal ASCII text
    buffer = io.BytesIO()
    np.savez_compressed(buffer, kspace=kspace_enc, adc_usage=adc_usage_enc)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')

    # The pulseq Siemens interpreter has a bug in the comment code leading to
    # errors if comments are longer than MAX_LINE_WIDTH = 256. We split the
    # data into chunks of 250 bytes to be on the safe side.
    with open(file_name, "a") as file:
        for i in range(0, len(encoded), 250):
            file.write(f"\n# {encoded[i:i+250]}")
        file.write("\n")


def extract_data_from_seq_file(
    file_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts kspace and adc_usage written with ``write_data_to_seq_file``.

    Parameters
    ----------
    file_name : str
        The name of the file the kspace was previously written to.

    Returns
    -------
    The original kspace and the adc_usage. There might be a  loss of precision
    because the kspace is written as 16 bit (half precision) floats and the
    usage as 8 bit integer (-128 to 127), this could be changed.
    """
    try:
        with open(file_name, "r") as file:
            # Find the last n lines that start with a '#'
            lines = file.readlines()
            
            if lines[-1][-1:] != '\n':
                lines[-1] = lines[-1] + '\n'
            
            n = len(lines)
            while n > 0 and lines[n-1][0] == '#':
                n -= 1
            if n == len(lines):
                raise ValueError("No data comment found at the end of the file")

            # Join the parts of the comment while removing "# " and "\n"
            encoded = "".join(line[2:-1] for line in lines[n:])
            # print(encoded)
            decoded = base64.b64decode(encoded, validate=True)

            data = np.load(io.BytesIO(decoded))
            kspace = np.cumsum(data["kspace"].astype(np.float32), 1).T
            adc_usage = data["adc_usage"].astype(np.int32)

            return torch.tensor(kspace), torch.tensor(adc_usage)
    except Exception as e:
        raise ValueError("Could not extract data from .seq") from e


def load_measurement(
    seq_file: str,
    seq_dat_file: str,
    wait_for_dat: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loads the seq data from a .seq file and the signal from a .seq.dat file.

    This function waits for the .seq.dat file if it doesn't exist yet and
    ``wait_for_dat = True``.

    Parameters
    ----------
    seq_file : str
        Name of the (path to the) .seq file
    seq_dat_file : str
        Name of the (path to the) .seq.dat file
    wait_for_dat : bool
        Specifies if this function should wait for the .seq.dat file or throw
        an error if it doesn't exist

    Returns
    -------
    (Samples, 4) tensor containing the kspace stored in the .seq file and a
    (Samples, Coils) tensor containing the signal (for all coils)
    """

    kspace, adc_usage = extract_data_from_seq_file(seq_file)

    if wait_for_dat:
        print("Waiting for TWIX file...", end="")
        while not os.path.isfile(seq_dat_file):
            time.sleep(0.2)
        print(" arrived!")

    data = np.loadtxt(seq_dat_file)
    data = data[:, 0] + 1j*data[:, 1]

    # .dat files contain additional samples we need to remove. This is probably
    # a bug in the TWIX to text file converter.
    #
    # These additional samples might be at the and of every shot or ADC block,
    # in which case a possible solution would be to store the subdivision in
    # the .seq file.
    #
    # Or maybe we can just fix it when exporting .seq files :D
    #
    # For now, we detect the number of samples in a single ADC readout and
    # assume 20 coils. Might not work for irregular readouts.

    # We assume that there are no exact zeros in the actual signal
    adc_length = np.where(np.abs(data) == 0)[0][0]
    data = data.reshape([-1, 20, adc_length + 4])

    # Remove additional samples and reshape into samples x coils
    signal = data.transpose([0, 2, 1])[:, :adc_length, :].reshape([-1, 20])

    if kspace.shape[0] != signal.shape[0]:
        print(
            f"WARNING: the kspace contains {kspace.shape[0]} samples but the "
            f"loaded signal has {signal.shape[0]}. They are either not for the"
            " same measurement, or something went wrong loading the data."
        )

    return kspace, adc_usage, torch.tensor(signal,dtype=torch.complex64)

def resize(tensor: torch.Tensor, new_size, mode='area'):
    # Functions expects batch x channels x (depth) x height x width
    if tensor.shape[-1] == 1:  # 2D, possible modes: 'area', 'bicubic'
        tensor_resized = tensor.squeeze().unsqueeze(0)
        tensor_resized = interpolate(tensor_resized, size=new_size[:2], mode=mode)
    else:  # 3D, possible modes: 'area', 'trilinear'
        tensor_resized = tensor.unsqueeze(0)
        tensor_resized = interpolate(tensor_resized, size=new_size, mode=mode)
    return tensor_resized.view(new_size)
