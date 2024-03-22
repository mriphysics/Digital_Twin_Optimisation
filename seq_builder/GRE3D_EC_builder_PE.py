from __future__ import annotations
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import MRzeroCore as mr0
import util


class GRE3D_EC:
    """Stores all parameters needed to create a 3D GRE sequence."""

    def __init__(self, adc_count: int, rep_count: int, part_count: int, Ndummies: int, R_accel: (int,int) = (1,1), dummies: (int,int) = (50,200)):
        """Initialize parameters with default values."""
        self.adc_count = adc_count*10
        self.event_count = adc_count*10 + dummies[0] + dummies[1] + 20*10 + 23*10 # DW: (ADC+PW+SP+RF+TRfill)
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
                # Extra events: pulse + winder + rewinder.
                rep = seq.new_rep(self.event_count+(oversampling-1)*self.adc_count)
    
                rep.pulse.angle = self.pulse_angles[ii]
                rep.pulse.phase = self.pulse_phases[ii]
                rep.pulse.usage = mr0.PulseUsage.EXCIT
    
                rep.event_time[:] = 0.1e-3/10
                
                # Split RF
                RF_len = 20*10
                rep.gradm[0:RF_len,:] = 0
    
                # Split prewinder
                rep.gradm[RF_len:RF_len+self.dummies[0], 0] = self.gradm_rewinder[ii] / self.dummies[0]
                rep.gradm[RF_len:RF_len+self.dummies[0], 1] = self.gradm_phase[ii] / self.dummies[0]
                rep.gradm[RF_len:RF_len+self.dummies[0], 2] = self.gradm_part[ii] / self.dummies[0]
                
                rep.gradm[RF_len+self.dummies[0]:self.adc_count+self.dummies[0]+RF_len, 0] = self.gradm_adc[ii]/(oversampling*10)  
    
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

class GRE3D_EC_PF:
    """Stores all parameters needed to create a 3D GRE sequence."""

    def __init__(self, adc_count: int, rep_count: int, part_count: int, Ndummies: int, pw_moment: int, spoiler_moment: float, R_accel: (int,int) = (1,1), dummies: (int,int) = (50,200)):
        """Initialize parameters with default values."""
        self.adc_count = int((pw_moment + adc_count/2)*10) # FG: fewer samples needed for PF acquisition; adc_count*10
        self.event_count = adc_count*10 + dummies[0] + dummies[1] + 20*10 + 23*10 # DW: (ADC+PW+SP+RF+TRfill)
        self.rep_count = pw_moment*2 # FG: maximum possible phase encoder defines number of accesible lines thus reps.
        self.part_count = part_count // R_accel[1]
        self.shots = 1
        self.R_accel = R_accel
        
        self.pw_moment = pw_moment

        # MODIFY FA HERE...
        self.pulse_angles = torch.full((self.rep_count*self.part_count, ), 15 * np.pi / 180)
        self.pulse_phases = torch.tensor(
            [util.phase_cycler(r, 117) for r in range(self.rep_count*self.part_count)])
        self.gradm_rewinder = torch.full((self.rep_count*part_count, ), -pw_moment-1) #FG: modify read prewinder for PF.
        
        self.gradm_phase = torch.arange(-self.rep_count//2+np.mod(self.rep_count//2,R_accel[0]), self.rep_count//2, R_accel[0]).repeat(self.part_count)
        self.gradm_part = torch.arange(-(self.part_count//2+np.mod(self.part_count//2,R_accel[1])), (self.part_count+1)//2, R_accel[1]).repeat_interleave(self.rep_count)
        self.gradm_adc = torch.full((self.rep_count*self.part_count, ), 1.0)
        self.gradm_spoiler = torch.full((self.rep_count*self.part_count, ), spoiler_moment * self.adc_count/10)
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
    
        
    def clone(self) -> GRE3D_EC_PF:
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
                # Extra events: pulse + winder + rewinder.
                rep = seq.new_rep(self.event_count+(oversampling-1)*self.adc_count)
    
                rep.pulse.angle = self.pulse_angles[ii]
                rep.pulse.phase = self.pulse_phases[ii]
                rep.pulse.usage = mr0.PulseUsage.EXCIT
    
                rep.event_time[:] = 0.1e-3/10
                
                # Split RF
                RF_len = 20*10
                rep.gradm[0:RF_len,:] = 0
    
                # Split prewinder
                rep.gradm[RF_len:RF_len+self.dummies[0], 0] = self.gradm_rewinder[ii] / self.dummies[0]
                rep.gradm[RF_len:RF_len+self.dummies[0], 1] = self.gradm_phase[ii] / self.dummies[0]
                rep.gradm[RF_len:RF_len+self.dummies[0], 2] = self.gradm_part[ii] / self.dummies[0]
                
                rep.gradm[RF_len+self.dummies[0]:self.adc_count+self.dummies[0]+RF_len, 0] = self.gradm_adc[ii]/(oversampling*10)  
    
                # Split spoiler
                rep.gradm[self.adc_count+self.dummies[0]+RF_len:self.adc_count+self.dummies[0]+RF_len+self.dummies[1], 0] = self.gradm_spoiler[ii] / self.dummies[1]
                rep.gradm[self.adc_count+self.dummies[0]+RF_len:self.adc_count+self.dummies[0]+RF_len+self.dummies[1], 1] = self.gradm_spoiler_phase[ii] / self.dummies[1]
                rep.gradm[self.adc_count+self.dummies[0]+RF_len:self.adc_count+self.dummies[0]+RF_len+self.dummies[1], 2] = self.gradm_spoiler_part[ii] / self.dummies[1]
                
                # Reset adc?
                rep.adc_phase = torch.zeros(rep.gradm.size(0),)
                rep.adc_usage = torch.zeros(rep.gradm.size(0),)                
                if ii < self.Ndummies:
                    rep.adc_usage[RF_len+self.dummies[0]:self.adc_count+self.dummies[0]+RF_len] = 0
                    rep.adc_phase[:] = 0
                else:
                    rep.adc_usage[RF_len+self.dummies[0]:self.adc_count+self.dummies[0]+RF_len] = 1
                    rep.adc_phase[:] = np.pi/2 - rep.pulse.phase
                
                # Reset event params?
                rep.event_count = rep.gradm.size(0)
                rep.event_time = torch.zeros(rep.gradm.size(0),)
                rep.event_time[:] = 0.1e-3/10
              
            seq_all.append(seq)
        
        return seq_all