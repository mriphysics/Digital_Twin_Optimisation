# -*- coding: utf-8 -*-
from types import SimpleNamespace
import sys
import numpy as np
import torch
import MRzeroCore as mr0
import util

sys.path.append("../scannerloop_libs")
from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_adc import make_adc as _make_adc
from pypulseq.make_delay import make_delay as _make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid as _make_trapezoid
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.add_gradients import add_gradients
from pypulseq.split_gradient import split_gradient
from pypulseq.opts import Opts

import ec_tools

from scipy.interpolate import interp1d

def rectify_flips(flips):
    flip_angle = flips.angle.cpu()
    flip_phase = flips.phase.cpu()

    if flips.angle < 0:
        flip_angle = -flips.angle
        flip_phase = flips.phase + np.pi
        flip_phase = torch.fmod(flip_phase, 2*np.pi)
    return flip_angle.item(), flip_phase.item()


# Modified versions of make_delay, make_adc and make_trapezoid that ensure that
# all events (and thus gradients) are on the gradient time raster. If they are
# not, the scanner crashes without hinting why

def make_delay(d: float) -> SimpleNamespace:
    """make_delay wrapper that rounds delay to the gradient time raster."""
    return _make_delay(round(d / 10e-6) * 10e-6)


def make_adc(num_samples: int, system: Opts = Opts(), dwell: float = 0, duration: float = 0, delay: float = 0,
             freq_offset: float = 0, phase_offset: float = 0) -> SimpleNamespace:
    """make_adc wrapper that modifies the delay such that the total duration
    is on the gradient time raster."""
    # TODO: the total duration might not be on the gradient raster. If a
    # sequence with optimized ADC durations fails the timing check, implement
    # this functions to round the timing as necessary.

    return _make_adc(
        num_samples, system, dwell, duration, delay,
        freq_offset, phase_offset
    )


def make_trapezoid(channel: str, amplitude: float = 0, area: float = None, delay: float = 0, duration: float = 0,
                   flat_area: float = 0, flat_time: float = -1, max_grad: float = 0, max_slew: float = 0,
                   rise_time: float = 0, system: Opts = Opts()) -> SimpleNamespace:
    """make_trapezoid wrapper that rounds gradients to the raster."""
    raster = system.grad_raster_time
    if delay != -1:
        delay = round(delay / raster) * raster
    if rise_time != -1:
        rise_time = round(rise_time / raster) * raster
    if flat_time != -1:
        flat_time = round(flat_time / raster) * raster
    if duration != -1:
        duration = round(duration / raster) * raster

    return _make_trapezoid(
        channel, amplitude, area, delay, duration, flat_area, flat_time,
        max_grad, max_slew, rise_time, system
    )


nonsel = 0
if nonsel==1:
    slice_thickness = 200*1e-3
else:
    slice_thickness = 8e-3
    



def interpolate_waveform(wave, dt_coarse=0.1e-3, dt_fine=1e-5):
# we need some way to go from the coarse time grid (0.1ms) to the gradient raster
# time (usually 10us) - doing it by linear interpolation here, but not sure if
# this is a good final solution
    n = len(wave)
    t_coarse = np.arange(0, n*dt_coarse, dt_coarse)
    t_fine = np.arange(0, n*dt_coarse, dt_fine)
    wave_fine = np.interp(t_fine, t_coarse, wave.cpu().numpy()) # linear interpolation
    
    return wave_fine


def pulseq_write_EPG_3D_EC(seq_param, path, FOV, plot_seq=False, num_slices=1, write_data=1, Ndummies_opt=0, smax=500, nonsel=0):
    
    # save pulseq definition
    slice_thickness = np.max([11e-3,5e-3*num_slices]) # hard coded here!
    FOV = FOV / 1000
    deltak = 1.0 / FOV # /(2*np.pi) before v.2.1.0

    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 80, "grad_unit": "mT/m", "max_slew": smax, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
    seq = Sequence(system)   
    
    seq.set_definition("FOV", [FOV, FOV, slice_thickness])
    seq.add_block(make_delay(5.0))   
    
    ############################### get gradient waveforms ###############################
    _, slew_x, waveform_x, waveformp_x = ec_tools.EC_perturbation_simple(seq_param, smax, Ndummies_opt, grad_dir=0, return_slew=True)
    _, slew_y, waveform_y, waveformp_y = ec_tools.EC_perturbation_simple(seq_param, smax, Ndummies_opt, grad_dir=1, return_slew=True)
    
    waveform_x = ec_tools.phys2moms(waveform_x) * deltak 
    waveform_y = ec_tools.phys2moms(waveform_y) * deltak
    
    waveform_x_interp = interpolate_waveform(waveform_x).reshape(len(seq_param), -1)
    waveform_y_interp = interpolate_waveform(waveform_y).reshape(len(seq_param), -1)

    ############################### !!!!!!!!!!!!! ###############################
    # TBD: the potentially optimized gradients during the RF pulse are ignored/discarded here.
    # Instead, a conventional slice selective pulse is used (trapezoidal slice selection and refocussing gradients)
    ############################### !!!!!!!!!!!!! ###############################

    for i,rep in enumerate(seq_param):
        adc_start = 0
        flip_angle,flip_phase = rectify_flips(rep.pulse)

        ############################### RF PULSE ###############################
        ## global pulse 
        RFdur=0
        if rep.pulse.usage == Seq.PulseUsage.UNDEF:
            RFdur=0
            if torch.abs(rep.pulse.angle) > 1e-8:
    
                RFdur = 2*1e-3
                kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                rf = make_block_pulse(**kwargs_for_block)
                
                seq.add_block(rf)  
                seq.add_block(make_delay(1e-4))
                RFdur += 1e-4
                                        
        elif (rep.pulse.usage == Seq.PulseUsage.EXCIT or
                rep.pulse.usage == Seq.PulseUsage.STORE):
            RFdur = 0
            if torch.abs(rep.pulse.angle) > 1e-8:
                use = "excitation"
                
                if nonsel:
                    RFdur = 2*1e-3 # hard-coded 2ms for now
                    kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                    rf_ex = make_block_pulse(**kwargs_for_block)
                    # seq.add_block(rf_ex)
                else:
                    # alternatively slice selective:
                    use = "excitation"
                    RFdur = 2*1e-3 - kwargs_for_opts['rf_dead_time'] # manually tweaked somehow to achieve real 2ms duration of slice selective pulse (including slice rewinder)
                    kwargs_for_sinc = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.15, "time_bw_product": 2, "phase_offset": flip_phase, "return_gz": True}
                    
                    rf_ex, gz, gzr= make_sinc_pulse(**kwargs_for_sinc)
                    
                    # build slice rephaser manually for better control over timing (does minimum time -> very steep traingle otherwise, whereas we have more time (prewinder duration, 0.5ms))
                    gzr = make_trapezoid(**{"channel": 'z', "system": system, "area": -0.5*gz.area, "duration": 5e-4, "delay": gz.rise_time + gz.flat_time + gz.fall_time + gz.delay})
                    
                    # concatenate all gz events to be able to later add everything in a single block
                    gz_all = add_gradients([gz, gzr], system)
                    
                    RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time + gz.delay   

        
        elif rep.pulse.usage == Seq.PulseUsage.REFOC:
            ###############################
            ### refocusing pulse
            use = "refocusing"
            
            RFdur = 0
            
            if torch.abs(rep.pulse.angle) > 1e-8:
                RFdur = 2*1e-3
                
                if nonsel:
                    kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                    rf_ref = make_block_pulse(**kwargs_for_block)
                    seq.add_block(rf_ref)         
                else:

                    kwargs_for_sinc = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flip_phase, "return_gz": True}
                    rf_ref, gz_ref,gzr = make_sinc_pulse(**kwargs_for_sinc)
                    seq.add_block(gzr)
                    seq.add_block(rf_ref, gz_ref)
                    seq.add_block(gzr)
                    RFdur = gz_ref.rise_time + gz_ref.flat_time + gz_ref.fall_time 
                    
        elif rep.pulse.usage == Seq.PulseUsage.FATSAT:
            ###############################
            ### fat saturation pulse
            use = "FatSat"
            RFdur = 6.120*1e-3
            dCurrFrequency = 123.2
            kwargs_for_gauss = {"flip_angle": 110*np.pi/180, "system": system, "slice_thickness": slice_thickness, "duration": RFdur, "freq_offset": -3.3*dCurrFrequency, "time_bw_product": 0.2, "apodization": 0.5, "return_gz": True} # "bandwidth": 200
            rf_ex, gz, gzr= make_gauss_pulse(**kwargs_for_gauss)
            seq.add_block(gzr)
            seq.add_block(rf_ex, gz)
            seq.add_block(gzr)          

        ############################### test: all together #####################
        gx_all_waveform = waveform_x_interp[i,:]
        gy_all_waveform = waveform_y_interp[i,:]        
        
        ##############
        # NEW IDEA: add entire waveform in one piece together with ADC, and shift ADC to right position via delay
        ##############
        kwargs_gx_all = {"channel": 'x', "system": system, "waveform": gx_all_waveform, "delay": rf_ex.delay  } 
        gx_all = make_arbitrary_grad(**kwargs_gx_all)
        
        kwargs_gy_all = {"channel": 'y', "system": system, "waveform": gy_all_waveform, "delay": rf_ex.delay  }
        gy_all = make_arbitrary_grad(**kwargs_gy_all)
        
        # ADC
        nADC = torch.sum(rep.adc_usage).item()
        if nADC > 0:
            adc_delay = 26*0.1e-3 - 4e-5 # start of read grad flat top (after interpolation) - manually empirically tuned via pulseq
            adc_duration = 32*0.1e-3 # HARD-CODED: adc duration
            kwargs_for_adc = {"num_samples": nADC, "duration": adc_duration, "delay":(adc_delay), "phase_offset": rf_ex.phase_offset - np.pi/4}
            adc = make_adc(**kwargs_for_adc)  
    
            if nonsel:
                seq.add_block(rf_ex, gx_all, gy_all, adc)
            else:
                seq.add_block(rf_ex, gz_all, gx_all, gy_all, adc)
        else:
            if nonsel:
                seq.add_block(rf_ex, gx_all, gy_all)
            else:
                seq.add_block(rf_ex, gz_all, gx_all, gy_all)

    
    passes, report = seq.check_timing()
    if not passes:
        print("WARNING: Timing check failed:")
        for line in report:
            print(line, end="")

    if plot_seq:
        seq.plot()
    seq.write(path)
    print(f'write seq file: {path}')
    
    if write_data:
        util.write_data_to_seq_file(seq_param, path)
        print('Added kspace & adc_usage information to the .seq file!')
        
        
def pulseq_write_EPG_3D_EC_EPI(seq_param, path, FOV, GIRF, plot_seq=False, num_slices=1, write_data=1, smax=500, nonsel=0):
    
    slice_thickness = np.max([11e-3,5e-3*num_slices]) # hard coded here!
    FOV = FOV / 1000
    deltak = 1.0 / FOV # /(2*np.pi) before v.2.1.0
    dt = 1e-5
        
    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 80, "grad_unit": "mT/m", "max_slew": smax, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
    seq = Sequence(system)   
    
    seq.set_definition("FOV", [FOV, FOV, slice_thickness])
    
    ############################### get gradient waveforms ###############################
    
    _, slew_x, slew_y, waveform_x, waveform_y, waveformp_x, waveformp_y = ec_tools.GIRF_perturbation(seq_param, FOV, GIRF, return_slew=True)
        
    # sort by repetitions    
    waveform_x = (ec_tools.phys2moms(waveform_x,FOV) * deltak).cpu() 
    waveform_y = (ec_tools.phys2moms(waveform_y,FOV) * deltak).cpu()
    
    rep = seq_param[0] # EPI: single repetition
    adc_block_idx = torch.nonzero((rep.adc_usage[1:] - rep.adc_usage[:-1])==1) + 1 # determine the start index of each adc block 

    adc_start = 0
    flip_angle,flip_phase = rectify_flips(rep.pulse)

    ############################### RF PULSE ###############################
    RFdur=0
                           
    if (rep.pulse.usage == mr0.PulseUsage.EXCIT or
            rep.pulse.usage == mr0.PulseUsage.STORE):
        RFdur = 0
        if torch.abs(rep.pulse.angle) > 1e-8:
            use = "excitation"
            
            if nonsel:
                RFdur = 2*1e-3 # hard-coded 2ms for now
                kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                rf_ex = make_block_pulse(**kwargs_for_block)
            else:
                # alternatively slice selective:
                use = "excitation"
                RFdur = 2*1e-3 - kwargs_for_opts['rf_dead_time'] # manually tweaked somehow to achieve real 2ms duration of slice selective pulse (including slice rewinder)
                kwargs_for_sinc = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.15, "time_bw_product": 2, "phase_offset": flip_phase, "return_gz": True}
                
                rf_ex, gz, gzr= make_sinc_pulse(**kwargs_for_sinc)
                
                # build slice rephaser manually for better control over timing (does minimum time -> very steep traingle otherwise, whereas we have more time (prewinder duration, 0.5ms))
                gzr = make_trapezoid(**{"channel": 'z', "system": system, "area": -0.5*gz.area, "duration": 5e-4, "delay": gz.rise_time + gz.flat_time + gz.fall_time + gz.delay})
                
                # concatenate all gz events to be able to later add everything in a single block
                gz_all = add_gradients([gz, gzr], system)
                
                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time + gz.delay     
    
    gx_RF = make_arbitrary_grad(**{"channel": 'x', "system": system, "waveform": waveform_x[:200], "delay": rf_ex.delay  })
    gy_RF = make_arbitrary_grad(**{"channel": 'y', "system": system, "waveform": waveform_y[:200], "delay": rf_ex.delay  })
    
    if nonsel:
        seq.add_block(rf_ex, gx_RF, gy_RF)
    else:
        seq.add_block(rf_ex, gx_RF, gy_RF, gz_all)

    ############################### PREWINDER ###############################
    gx_pre = make_arbitrary_grad(**{"channel": 'x', "system": system, "waveform": waveform_x[200:adc_block_idx[0]], "delay": 0  })
    gy_pre = make_arbitrary_grad(**{"channel": 'y', "system": system, "waveform": waveform_y[200:adc_block_idx[0]], "delay": 0  })
    seq.add_block(gx_pre, gy_pre)
    
    ############################### EPI TRAIN ###############################
    for jj in range(len(adc_block_idx)):
        start_idx = adc_block_idx[jj]
        
        if jj < len(adc_block_idx)-1:
            end_idx = adc_block_idx[jj+1]
        else:
            end_idx = len(waveform_x)
        
        gx_r = make_arbitrary_grad(**{"channel": 'x', "system": system, "waveform": waveform_x[start_idx:end_idx], "delay": 0  })
        gy_r = make_arbitrary_grad(**{"channel": 'y', "system": system, "waveform": waveform_y[start_idx:end_idx], "delay": 0  })
        
        adc_delay = 0
        nADC = adc_block_idx.shape[0]
        adc_duration = nADC*dt 
        adc = make_adc(**{"num_samples": nADC, "duration": adc_duration, "delay":adc_delay, "phase_offset": rf_ex.phase_offset - np.pi/4})  
        
        seq.add_block(gx_r, gy_r, adc)
        
    gx_dt2 = make_arbitrary_grad(**{"channel": 'x', "system": system, "waveform": waveform_x[len(waveform_x)-8:], "delay": 0  })
    gy_dt2 = make_arbitrary_grad(**{"channel": 'y', "system": system, "waveform": waveform_y[len(waveform_x)-8:], "delay": 0  })
    seq.add_block(gx_dt2, gy_dt2)
    
    passes, report = seq.check_timing()
    if not passes:
        print("WARNING: Timing check failed:")
        for line in report:
            print(line, end="")

    if plot_seq:
        seq.plot()
    seq.write(path)
    print(f'write seq file: {path}')
    
    if write_data:
        util.write_data_to_seq_file(seq_param, path)
        print('Added kspace & adc_usage information to the .seq file!')

def append_header(path):
    # append version and definitions
    if sys.platform != 'linux':
        try:
            with open(r"\\141.67.249.47\MRTransfer\mrzero_src\.git\ORIG_HEAD") as file:
                git_version = file.read()
        except:
            git_version = ''
    with open(path, 'r') as fin:
        lines = fin.read().splitlines(True)

    updated_lines = []
    updated_lines.append("# Pulseq sequence file\n")
    updated_lines.append("# Created by MRIzero/IMR/GPI pulseq converter\n")
    if sys.platform != 'linux':
        updated_lines.append('# MRZero Version: 0.5, git hash: ' + git_version)
    if sys.platform == 'linux':
        updated_lines.append("# experiment_id: "+path.split('/')[-2]+"\n")
    else:
        updated_lines.append("# experiment_id: "+path.split('\\')[-2]+"\n")
    updated_lines.append("# path: " + path + "\n")
    updated_lines.append("\n")
    
    updated_lines.extend(lines[3:])

    with open(path, 'w') as fout:
        fout.writelines(updated_lines)
