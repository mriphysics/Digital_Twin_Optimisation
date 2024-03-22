from __future__ import annotations
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import MRzeroCore as mr0
import util


class GRE3D_EC:
    """Stores all parameters needed to create a 3D GRE sequence."""

    def __init__(self, adc_count: int, rep_count: int, part_count: int, Ndummies: int, R_accel: (int,int) = (1,1), dummies: (int,int) = (5,20)):
        """Initialize parameters with default values."""
        self.adc_count = adc_count
        self.event_count = adc_count + dummies[0] + dummies[1] + 20 + 23 # DW: (ADC+PW+SP+RF+TRfill)
        self.rep_count = rep_count // R_accel[0]
        self.part_count = part_count // R_accel[1]
        self.shots = 1
        self.R_accel = R_accel

        # MODIFY FA HERE...
        self.pulse_angles = torch.full((self.rep_count*self.part_count, ), 15 * np.pi / 180)
        self.pulse_phases = torch.tensor(
            [util.phase_cycler(r, 117) for r in range(self.rep_count*self.part_count)])
        self.gradm_rewinder = torch.full((rep_count*part_count, ), -adc_count/2-1)
        
        self.gradm_phase = torch.arange(-rep_count//2+np.mod(rep_count//2,R_accel[0]), rep_count//2, R_accel[0]).repeat(self.part_count)
        self.gradm_part = torch.arange(-(part_count//2+np.mod(part_count//2,R_accel[1])), (part_count+1)//2, R_accel[1]).repeat_interleave(self.rep_count)
        self.gradm_adc = torch.full((rep_count*part_count, ), 1.0)
        self.gradm_spoiler = torch.full((rep_count*part_count, ), 1.5 * adc_count)
        self.gradm_spoiler_phase = -self.gradm_phase
        self.gradm_spoiler_part = -self.gradm_part
        self.TE = 0
        self.TR = 0
        
        self.relaxation_time = torch.tensor(1e-5)
        
        self.dummies = dummies # (splitPrewinder, splitSpoiler) for split gradient optimization.
        
        self.Ndummies = Ndummies # To reach a steady-state (defined above).
        
        
    def linearEncoding(self, adc_count: int, rep_count: int, part_count: int) -> GRE3D_EC:
        rep_count = rep_count-self.Ndummies
        tmp = torch.zeros([self.Ndummies]) # DUMMY TRs - SET PE TO ZERO        
        self.gradm_phase = torch.cat((tmp,torch.arange(-rep_count//2+np.mod(rep_count//2,self.R_accel[0]), rep_count//2, self.R_accel[0]).repeat(self.part_count)))
        self.gradm_part = torch.arange(-(part_count//2+np.mod(part_count//2,self.R_accel[1])), (part_count+1)//2, self.R_accel[1]).repeat_interleave(self.rep_count)
        self.gradm_spoiler_phase = -self.gradm_phase
        self.gradm_spoiler_part = -self.gradm_part
        
    def centricEncoding(self, adc_count: int, rep_count: int, part_count: int) -> GRE3D_EC:
        
        # permutation vector 
        def permvec(x) -> np.ndarray:
            permvec = np.zeros((x,),dtype=int) 
            permvec[0] = 0
            for i in range(1,int(x/2)+1):
                permvec[i*2-1] = (-i)
                if i < x/2:
                    permvec[i*2] = i
            return permvec+x//2  
        
        tmp = torch.arange(-rep_count//2+np.mod(rep_count//2,self.R_accel[0]), rep_count//2, self.R_accel[0])
        self.gradm_phase = tmp[permvec(self.rep_count)].repeat(self.part_count)
        tmp = torch.arange(-(part_count//2+np.mod(part_count//2,self.R_accel[1])), (part_count+1)//2, self.R_accel[1])
        self.gradm_part = tmp[permvec(self.part_count)].repeat_interleave(self.rep_count)
        self.gradm_spoiler_phase = -self.gradm_phase
        self.gradm_spoiler_part = -self.gradm_part
    
    
    def spiralEncoding(self, spiral_elongation = 0, alternating = False) -> GRE3D_EC:
        """Create spiral encoding in y and z direction.""" 
        # permutation vector 
        def permvec(x) -> np.ndarray:
            permvec = np.zeros((x,),dtype=int) 
            permvec[0] = 0
            for i in range(1,int(x/2)+1):
                permvec[i*2-1] = (-i)
                if i < x/2:
                    permvec[i*2] = i
            return permvec+x//2   
        
        a, b = torch.meshgrid(self.gradm_phase[:self.rep_count],self.gradm_part[::self.rep_count])
        reordering_y = []
        reordering_z = []
        size_y = a.shape[0]
        size_z = a.shape[1]
        
        corr = 0
        if spiral_elongation == 0:
            Iy = 1 # Number of first encoding line in y direction
            Iz = 1 # Number of first encoding line in z directio
            pos_lin = a.shape[0]//2 # Position in y direction
            pos_par = a.shape[1]//2 # Position in z direction
        elif spiral_elongation > 0:
            Iy = int(np.ceil(np.abs(spiral_elongation)*size_y)) + 1
            Iz = 1
            pos_lin = a.shape[0]//2+int(np.ceil(Iy/2))-1 # Position in y direction
            pos_par = a.shape[1]//2 # Position in z direction
        elif spiral_elongation < 0:
            Iy = 1
            Iz = int(np.ceil(np.abs(spiral_elongation)*size_z))
            pos_lin = a.shape[0]//2 # Position in y direction
            pos_par = a.shape[1]//2-int(np.ceil(Iz/2)) # Position in z direction
            for jj in range(0,Iz):
                #print(jj)
                reordering_y.append(a[pos_lin,pos_par+jj])
                reordering_z.append(b[pos_lin,pos_par+jj])
            pos_par += Iz
            corr = 1
        
        sign = 1
        Iy = Iy
        Iz = Iz+corr
                
        while (Iy < size_y) or (Iz < size_z) or len(reordering_y) < size_y*size_z:
            pos_lin = min(pos_lin,size_y-1)
            pos_par = min(pos_par,size_z-1)
            if Iz <= a.shape[1]:
                for ii in range(0,min(Iy,size_y)):
                    #print(ii)
                    reordering_y.append(a[pos_lin-sign*ii,pos_par])
                    reordering_z.append(b[pos_lin-sign*ii,pos_par])
            else:
                Iz = min(Iz,size_z)
            pos_lin -= sign*min(Iy,size_y-1)
            
            if Iy <= size_y:
                for jj in range(0,Iz):
                    #print(jj)
                    reordering_y.append(a[pos_lin,pos_par-sign*jj])
                    reordering_z.append(b[pos_lin,pos_par-sign*jj])
            else:
               Iy = min(Iy,size_y) 
            Iy += 1
            pos_par -= sign*min(Iz,size_z-1)
            Iz += 1
            # print(j)
            # print(i)
            sign *= -1

        num_perm = max(int(np.ceil(spiral_elongation*size_y))-1,int(np.ceil(-spiral_elongation*size_z)))+1
        perm = permvec(num_perm) 
        
        self.gradm_phase = torch.tensor(reordering_y)
        self.gradm_part = torch.tensor(reordering_z)
        
        if alternating:
            self.gradm_phase[:num_perm] = self.gradm_phase[perm]
            self.gradm_part[:num_perm] = self.gradm_part[perm]
        
        self.gradm_spoiler_phase = -self.gradm_phase
        self.gradm_spoiler_part = -self.gradm_part
        cmap = plt.cm.get_cmap('rainbow')
        
        plt.plot(self.gradm_part,self.gradm_phase); plt.xlabel('z'); plt.ylabel('y');plt.title('Spiral elongation = ' + str(spiral_elongation))
        for i in range(num_perm):
            plt.plot(self.gradm_part[i], self.gradm_phase[i],'.', c=cmap(i / num_perm))
        plt.show()
        
    def clone(self) -> GRE3D_EC:
        """Create a copy with cloned tensors."""
        clone = GRE3D_EC(self.adc_count, self.rep_count, self.part_count)

        clone.pulse_angles = self.pulse_angles.clone()
        clone.pulse_phases = self.pulse_phases.clone()
        clone.gradm_rewinder = self.gradm_rewinder.clone()
        clone.gradm_phase = self.gradm_phase.clone()
        clone.gradm_part = self.gradm_part.clone()
        clone.gradm_adc = self.gradm_adc.clone()
        clone.gradm_spoiler = self.gradm_spoiler.clone()
        clone.gradm_spoiler_phase = self.gradm_spoiler_phase.clone()
        clone.gradm_spoiler_part = self.gradm_spoiler_part.clone()
        clone.relaxation_time = self.relaxation_time.clone()

        return clone

    def generate_sequence(self, oversampling = 1) -> mr0.Sequence:
        """Generate a GRE sequence based on the given parameters."""
        
        seq_all = []

        for shot in range(self.shots): 
            seq = mr0.Sequence()
            
            for ii in torch.arange(shot,self.part_count*self.rep_count,self.shots):
                # Extra events: pulse + winder + rewinder
                rep = seq.new_rep(self.event_count+(oversampling-1)*self.adc_count)
    
                rep.pulse.angle = self.pulse_angles[ii]
                rep.pulse.phase = self.pulse_phases[ii]
                rep.pulse.usage = mr0.PulseUsage.EXCIT
    
                rep.event_time[:] = 0.1e-3
                
                # Split RF
                RF_len = 20
                rep.gradm[0:RF_len,:] = 0
    
                # Split prewinder
                rep.gradm[RF_len:RF_len+self.dummies[0], 0] = self.gradm_rewinder[ii] / self.dummies[0]
                rep.gradm[RF_len:RF_len+self.dummies[0], 1] = self.gradm_phase[ii] / self.dummies[0]
                rep.gradm[RF_len:RF_len+self.dummies[0], 2] = self.gradm_part[ii] / self.dummies[0]
                
                rep.gradm[RF_len+self.dummies[0]:self.adc_count+self.dummies[0]+RF_len, 0] = self.gradm_adc[ii]/oversampling  
    
                # Split spoiler
                rep.gradm[self.adc_count+self.dummies[0]+RF_len:self.adc_count+self.dummies[0]+RF_len+self.dummies[1], 0] = self.gradm_spoiler[ii] / self.dummies[1]
                rep.gradm[self.adc_count+self.dummies[0]+RF_len:self.adc_count+self.dummies[0]+RF_len+self.dummies[1], 1] = self.gradm_spoiler_phase[ii] / self.dummies[1]
                rep.gradm[self.adc_count+self.dummies[0]+RF_len:self.adc_count+self.dummies[0]+RF_len+self.dummies[1], 2] = self.gradm_spoiler_part[ii] / self.dummies[1]
    
                if ii < self.Ndummies:
                    rep.adc_usage[RF_len+self.dummies[0]:self.adc_count+self.dummies[0]+RF_len] = 0
                    rep.adc_phase[:] = 0
                else:
                    rep.adc_usage[RF_len+self.dummies[0]:self.adc_count+self.dummies[0]+RF_len] = 1
                    rep.adc_phase[:] = np.pi/2 - rep.pulse.phase
                                
            seq_all.append(seq)

        return seq_all        
    
    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_name) -> GRE3D_EC:
        with open(file_name, 'rb') as file:
            return pickle.load(file)


def plot_optimization_progress(
    reco: torch.Tensor, reco_target: torch.Tensor,
    params: GRE3D_EC, params_target: GRE3D_EC,
    kspace_trajectory: list[torch.Tensor], loss_history: list[float],
    figsize: tuple[float, float] = (10, 10), dpi: float = 180
) -> np.ndarray:
    """
    Plot a picture containing the most important sequence properties.

    This function also returns the plotted image as array for gif creation.
    """
    plt.figure(figsize=figsize)
    reco_max = max(np.abs(util.to_numpy(reco[:, :, 0])).max(),
                   np.abs(util.to_numpy(reco_target[:, :, 0])).max())
    plt.subplot(3, 2, 1)
    plt.imshow(np.abs(util.to_numpy(reco[:, :, 0])), vmin=0, vmax=reco_max)
    plt.colorbar()
    plt.title("Reco")
    plt.subplot(3, 2, 3)
    plt.imshow(np.abs(util.to_numpy(reco_target[:, :, 0])), vmin=0, vmax=reco_max)
    plt.colorbar()
    plt.title("Target")

    plt.subplot(3, 2, 2)
    plt.plot(np.abs(util.to_numpy(params.pulse_angles)) * 180 / np.pi, '.')
    plt.plot(util.to_numpy(params_target.pulse_angles) * 180 / np.pi, '.', color='r')
    plt.title("Flip Angles")
    plt.ylim(bottom=0)
    plt.subplot(3, 2, 4)
    plt.plot(np.mod(np.abs(util.to_numpy(params.pulse_phases)) * 180 / np.pi, 360), '.')
    plt.plot(np.mod(np.abs(util.to_numpy(params_target.pulse_phases)) * 180 / np.pi, 360), '.', color='r')
    plt.title("Phase")

    plt.subplot(3, 2, 5)
    plt.plot(loss_history)
    plt.yscale('log')
    plt.grid()
    plt.title("Loss Curve")

    plt.subplot(3, 2, 6)
    for i, rep_traj in enumerate(kspace_trajectory):
        kx = util.to_numpy(rep_traj[:, 0]) / (2*np.pi)
        ky = util.to_numpy(rep_traj[:, 1]) / (2*np.pi)
        plt.plot(kx, ky, c=cm.rainbow(i / len(kspace_trajectory)))
        plt.plot(kx, ky, 'k.')
    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")
    plt.grid()

    img = util.current_fig_as_img(dpi)
    plt.show()
    return img

class EPI2D_EC:
    """Stores all parameters needed to create a 2D EPI sequence."""

    def __init__(self, adc_count: int, rep_count: int):
        
        self.adc_count = adc_count
        self.rep_count = 1
        self.event_count = 4720+656 # HARD-CODED: 200 (RF) + 500 (prew) + len(p2r_x) + len(RF2p_x) + len(newx) + 2
        self.part_count = 1
        self.shots = 1
        ros     = 1
        self.TE = 0
        self.TR = 0

        self.pulse_angles = torch.tensor(90*np.pi/180)
        self.pulse_phases = torch.tensor(0)

        self.gradm_xpre = -adc_count/2
        self.gradm_ypre = -adc_count/2

        self.gradm_x = torch.ones((adc_count*ros+1,adc_count)) * 1/ros # FG: inserted +1 here to have all ADC samples during flat top.
        self.gradm_x[:,1::2] = -self.gradm_x[:,1::2]
        self.gradm_x[-1,:] = 0
        
        self.gradm_y = torch.zeros((adc_count*ros+1,adc_count))
        self.gradm_y[-1,:] = 1


    def clone(self) -> EPI2D_EC:
        """Create a copy with cloned tensors."""
        clone = EPI2D_EC(self.adc_count, self.rep_count)

        clone.pulse_angles = self.pulse_angles.clone()
        clone.pulse_phases = self.pulse_phases.clone()
        clone.gradm_x      = self.gradm_x.clone()
        clone.gradm_xpre   = self.gradm_xpre.clone()
        clone.gradm_y      = self.gradm_y.clone()
        clone.gradm_ypre   = self.gradm_ypre.clone()

        return clone

    def generate_sequence(self, oversampling = 1) -> mr0.Sequence:
        """Generate a EPI sequence based on the given parameters."""
        
        def insert_ramp_periods(original_tensor,nsteps):
            new_tensor = []
            
            prev_value = original_tensor[0].item()
            new_tensor.append(prev_value)
            
            for current_value in original_tensor[1:]:
                current_value = current_value.item()
                
                if current_value != prev_value:
                    if current_value > 0 and prev_value == 0:
                        ramp_values = torch.linspace(0, current_value, nsteps)[1:-1]
                    elif current_value == 0 and prev_value > 0:
                        ramp_values = torch.linspace(prev_value, 0, nsteps)[1:-1]
                    elif current_value < 0 and prev_value == 0:
                        ramp_values = torch.linspace(0, current_value, nsteps)[1:-1]
                    elif current_value == 0 and prev_value < 0:
                        ramp_values = torch.linspace(prev_value, 0, nsteps)[1:-1]
                    else:
                        ramp_values = []
                    
                    new_tensor.extend(ramp_values)
                    new_tensor.append(current_value)
                else:
                    new_tensor.append(current_value)
                
                prev_value = current_value
            
            return torch.tensor(new_tensor)       
        
        seq_all = []

        nsteps = 6 # CAN CHANGE THIS TO MODIFY INITIAL SLEW RATE

        for shot in range(self.shots): 
            seq = mr0.Sequence()
            
            for ii in torch.arange(shot,self.part_count*self.rep_count,self.shots):
               
                rep = seq.new_rep(self.event_count+(oversampling-1)*self.adc_count)
                
                rep.event_time[:] = 10e-6                
                
                rep.pulse.angle = self.pulse_angles
                rep.pulse.phase = self.pulse_phases
                rep.pulse.usage = mr0.PulseUsage.EXCIT
                                
                # Prepare RO period waveforms.
                tmpx = self.gradm_x.permute((1,0)).reshape((-1,))
                tmpy = self.gradm_y.permute((1,0)).reshape((-1,)) * (1/(nsteps-1)) # To account for larger triangle area with ramps.                
                newx = insert_ramp_periods( tmpx, nsteps)
                newy = insert_ramp_periods(-tmpy, nsteps)
                
                # Define RF period.
                RF_len = 200 # RF AT START
                rep.gradm[:RF_len,:] = 0
                
                # Define prewinder - analytical attempts of taking ramping moments into account.
                pw_len = 500
                px_tmp = (self.gradm_xpre - nsteps/2*(rep.gradm[RF_len,0]+newx[0])) / (pw_len + nsteps - 2)
                py_tmp = (self.gradm_ypre - nsteps/2*(0+0)) / (pw_len + nsteps - 2) # 0+0 corresponds to initial and final grad value
                
                # Deal with ramp before prewinder.
                RF2p_x = torch.linspace(rep.gradm[RF_len,0], px_tmp, nsteps)[1:-1]
                RF2p_y = torch.linspace(rep.gradm[RF_len,1], py_tmp, nsteps)[1:-1]
                              
                rep.gradm[RF_len:RF_len+len(RF2p_x),0] = RF2p_x
                rep.gradm[RF_len:RF_len+len(RF2p_y),1] = RF2p_y
                 
                rep.gradm[RF_len+len(RF2p_x):RF_len+len(RF2p_x)+pw_len,0] = px_tmp
                rep.gradm[RF_len+len(RF2p_x):RF_len+len(RF2p_x)+pw_len,1] = py_tmp
                
                # Deal with ramp after prewinder.
                p2r_x = torch.linspace(rep.gradm[RF_len+len(RF2p_x):RF_len+len(RF2p_x)+pw_len,0][-1], newx[0], nsteps)[1:-1]
                p2r_y = torch.linspace(rep.gradm[RF_len+len(RF2p_x):RF_len+len(RF2p_x)+pw_len,1][-1], newy[0], nsteps)[1:-1]                   

                rep.gradm[RF_len+len(RF2p_x)+pw_len:RF_len+len(RF2p_x)+pw_len+len(p2r_x),0] = p2r_x
                rep.gradm[RF_len+len(RF2p_x)+pw_len:RF_len+len(RF2p_x)+pw_len+len(p2r_y),1] = p2r_y

                RO_len = len(newx) # X and Y have same no. elements.
                rep.gradm[RF_len+len(RF2p_x)+pw_len+len(p2r_x):RF_len+len(RF2p_x)+pw_len+len(p2r_x)+RO_len,0] =  newx
                rep.gradm[RF_len+len(RF2p_x)+pw_len+len(p2r_x):RF_len+len(RF2p_x)+pw_len+len(p2r_x)+RO_len,1] = -newy
                
                indices_p = torch.nonzero(newx ==  1)
                indices_n = torch.nonzero(newx == -1)  
                                
                rep.adc_usage[RF_len+len(RF2p_x)+pw_len+len(p2r_x)+indices_p-0] = 1 # FG: odd/even line discrepancy?
                rep.adc_usage[RF_len+len(RF2p_x)+pw_len+len(p2r_x)+indices_n-1] = 1 # Use 0
                
            seq_all.append(seq)
        
        return seq_all        
    
    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_name) -> GRE3D_EC:
        with open(file_name, 'rb') as file:
            return pickle.load(file)