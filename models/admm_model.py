'''
Code Adapted from https://github.com/Waller-Lab/LenslessLearning
'''


import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from helper_functions import *
from models.admm_rgb import admm
from models.admm_helper import *

#ADMM net
class ADMM_Net(nn.Module):
    
    def __init__(self,batch_size,h, iterations, cuda_device, learning_options={'learned_vars':[]},
     le_admm_s=False,denoise_model=[]):
        """
        constructor to initialize the ADMM network.

        Args:
            batch_size (int): Batch size
            h (np.array): PSF of the imaging system size (270,480)
            iterations (int): number of unrolled iterations
            learning_options (dict, optional): variables to be learned Defaults to {'learned_vars':[]}.
            cuda_device (str, optional): device {cuda or cpu}. Defaults to torch.device('cpu').
            le_admm_s (bool, optional): Turn on if using Le-ADMM*, otherwise should be set to False. Defaults to False.
            denoise_model (list, optional): model to use as a learnable regularizer. Defaults to [].
        """
        super(ADMM_Net, self).__init__()        
        #number of unrolled iterations
        
        self.iterations=iterations
        #batch size
        self.batch_size=batch_size
        #using autotune
        self.autotune=False
        #real data or the simulated data
        self.realdata=True
        #print ADMM variables
        self.printstats=False
        
        # add noise ( only if simulated data)
        self.addnoise=False
        # noise standard deviation
        self.noise_std=0.05
        self.cuda_device=cuda_device
        self.l_admm_s=le_admm_s
        if le_admm_s==True:
            self.Denoiser=denoise_model.to(cuda_device)
           
        #learned structure options
        self.learning_options=learning_options 
        
        #initialize constants
        self.DIMS0 = h.shape[0]  # Image Dimensions
        self.DIMS1 = h.shape[1]  # Image Dimensions

        self.PAD_SIZE0 = int((self.DIMS0)//2)  # Pad size
        self.PAD_SIZE1 = int((self.DIMS1)//2)  # Pad size
        #initialize variables
        self.initialize_learned_variables(learning_options)
        #psf
        self.h_var=torch.nn.Parameter(torch.tensor(h,dtype=torch.float32, device=self.cuda_device),requires_grad=False)
        self.h_padded=F.pad(self.h_var,(self.PAD_SIZE1,self.PAD_SIZE1,self.PAD_SIZE0,self.PAD_SIZE0),'constant',0)
        
        # shift the zero frequency component from center to the corner
        self.h_center_right=torch.fft.fftshift(self.h_padded)
        
        # compute the 2D discrete Fourier transform
        self.H=torch.fft.fft2(self.h_center_right)
        
        self.Hconj_new=torch.conj(self.H)
        self.HtH=self.H*self.Hconj_new
        self.HtH=self.HtH.real
        #LtL is the sparsifying transformation 
        self.LtL = torch.nn.Parameter(torch.tensor(make_laplacian(self), dtype=torch.float32, device=self.cuda_device),
                                      requires_grad=False)
        self.resid_tol =  torch.tensor(1.5, dtype= torch.float32, device=self.cuda_device)
        # change of penalizing factor
        self.mu_inc = torch.tensor(1.2, dtype = torch.float32, device=self.cuda_device)
        self.mu_dec = torch.tensor(1.2, dtype = torch.float32, device=self.cuda_device)
        
    def initialize_learned_variables(self,learning_options):
            #mu are the scaler penalty parameters  for each iterations
            if 'mus' in learning_options['learned_vars']:  
                #initialize to small value i.e 1e-04 for each iteration
                self.mu1= torch.nn.Parameter(torch.tensor(np.ones(self.iterations)*1e-4, dtype = torch.float32,device=self.cuda_device))
                self.mu2= torch.nn.Parameter(torch.tensor(np.ones(self.iterations)*1e-4, dtype = torch.float32,device=self.cuda_device))
                self.mu3= torch.nn.Parameter(torch.tensor(np.ones(self.iterations)*1e-4, dtype = torch.float32,device=self.cuda_device))
            else:
                #initialize to small value but doesn't make it learnable
                self.mu1=  torch.ones(self.iterations, dtype = torch.float32, device=self.cuda_device)*1e-4
                self.mu2=  torch.ones(self.iterations, dtype = torch.float32, device=self.cuda_device)*1e-4
                self.mu3 = torch.ones(self.iterations, dtype = torch.float32, device=self.cuda_device)*1e-4
            
            if "tau" in learning_options['learned_vars']: # tau parameter 
                self.tau = torch.nn.Parameter(torch.tensor(np.ones(self.iterations)*2e-4,dtype=torch.float32,device=self.cuda_device))
            
                #  initialize to small value
            else:
                self.tau=torch.ones(self.iterations, dtype = torch.float32,device=self.cuda_device)*2e-3
        
    def forward(self, inputs):  
        self.batch_size=inputs.shape[0]
        #mu and tau parameters
        self.mu_vals=torch.stack([self.mu1,self.mu2,self.mu3,self.tau])
        self.admmstats = {'dual_res_s': [], 'dual_res_u': [], 'dual_res_w': [], 
             'primal_res_s': [], 'primal_res_u': [], 'primal_res_w': [],
             'data_loss': [], 'total_loss': []}
        if self.autotune==True:
            self.mu_auto_list= {'mu1': [], 'mu2': [], 'mu3': []}
        y = inputs.to(self.cuda_device)
        Cty = pad_zeros_torch(self, y).to(self.cuda_device) #(Ctx)
        CtC = pad_zeros_torch(self, torch.ones_like(y,device=self.cuda_device))     # Zero padded ones with the shape of input y (CtC)
            
        in_vars = [] 
        in_vars1 = []
        in_vars2 = []
        Hsk_list = []
        a2k_1_list=[]
        a2k_2_list= []
        
        sk = torch.zeros_like(Cty, dtype = torch.float32,device=self.cuda_device)
        # larange multipliers
        alpha1k = torch.zeros_like(Cty, dtype = torch.float32,device=self.cuda_device)
        alpha3k = torch.zeros_like(Cty, dtype = torch.float32,device=self.cuda_device)
        #Hxk from the paper for the vkp
        Hskp = torch.zeros_like(Cty, dtype = torch.float32,device=self.cuda_device)
        
        # if learnable addam is set true( used for Le-ADMM-*)
        if self.l_admm_s == True:
            # use of U-net as a denoiser after the iteration ie. sk is feed towards the U-net
            Lsk_init, mem_init = self.Denoiser.forward(sk)
            # set aplha2k as the size of the output after the denoiser , drop the alpha2k
            alpha2k = torch.zeros_like(Lsk_init, dtype = torch.float32,  device=self.cuda_device)
        
        else:
            alpha2k_1 = torch.zeros_like(sk[:,:,:-1,:], dtype = torch.float32,device=self.cuda_device)  
            alpha2k_2 = torch.zeros_like(sk[:,:,:,:-1], dtype = torch.float32,device=self.cuda_device)
            
            a2k_1_list.append(alpha2k_1)
            a2k_2_list.append(alpha2k_2)                             
        mu_auto = torch.stack([self.mu1[0], self.mu2[0], self.mu3[0], self.tau[0]])
        in_vars.append(torch.stack([sk, alpha1k, alpha3k, Hskp]))
        
        for i in range(0,self.iterations):
            
           out_vars, a_out1, a_out2, _ , symm, admmstats = admm(self, in_vars[-1], 
                                                              a2k_1_list[-1], a2k_2_list[-1], CtC, Cty, [], i, y)
           in_vars.append(out_vars)
           a2k_1_list.append(a_out1)
           a2k_2_list.append(a_out2)
           
        x_out = crop(self, in_vars[-1][0])
        x_outn = normalize_image(x_out)
        self.in_list = in_vars
        return x_outn
                
        
        
        
        
