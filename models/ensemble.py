import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, admm_model, denoise_model):
        super(EnsembleModel, self).__init__()
        self.admm_model = admm_model
        self.denoise_model = denoise_model
        
    def forward(self, x):
        
        admm_output = self.admm_model(x)
        #pad input image to be a multiple of window_size (pad to the right and bottom)
        
        # _, _, h_old, w_old = admm_output.size()
        # window_size=8
        # h_pad = (h_old // window_size + 1) * window_size - h_old
        # w_pad = (w_old // window_size + 1) * window_size - w_old
        # img_lq = torch.cat([admm_output, torch.flip(admm_output, [2])], 2)[:, :, :h_old + h_pad, :]
        # img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        # final_output = self.denoise_model(admm_output) 
        # # return to the orginal shape by cropping
        # final_output=final_output[:,:,:h_old,:w_old]
        
        final_output = self.denoise_model(admm_output) 
        return final_output
    
    def to(self, indevice):
        self = super().to(indevice)
        self.admm_model.to(indevice)
        self.admm_model.h_var.to(indevice)
        #self.admm_model.h_zeros.to(indevice)
        #self.admm_model.h_complex.to(indevice)
        self.admm_model.LtL.to(indevice)
        return self
    