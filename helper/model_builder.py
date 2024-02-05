import torch,sys
import numpy as np
from skimage.transform import resize

sys.path.append('/home/arpanp/diffuser_cam_vit/models')
from models.admm_model import ADMM_Net
from utils import *

def build_model(batch_size,iterations,device):
    
    # Load PSF
    path_diffuser ='sample_images/psf.tiff'
    psf_diffuser = load_psf_image2(path_diffuser, downsample=1, rgb= False)

    ds = 4   # Amount of down-sampling.  Must be set to 4 to use dataset images 
    psf_diffuser = np.sum(psf_diffuser,2)
    h = resize(psf_diffuser, (psf_diffuser.shape[0]//ds,psf_diffuser.shape[1]//ds), mode='constant', anti_aliasing=True)
    var_options = {'plain_admm': [],
               'mu_and_tau': ['mus', 'tau'],
              }
    learning_options_admm = {'learned_vars': var_options['mu_and_tau']}
    
    model= ADMM_Net(batch_size = batch_size, h = h, iterations = iterations, 
                           learning_options = learning_options_admm, cuda_device =device)
    
    return model 
    
    
