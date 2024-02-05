import numpy as np 
import torch
import torch.nn.functional as F
from torchmetrics.functional import image_gradients

#calculates the gradient of the image
def L_tf(a):
    xdiff = a[:,:, 1:, :]-a[:,:, :-1, :]
    ydiff = a[:,:, :, 1:]-a[:,:, :, :-1]
    return -xdiff, -ydiff

####### Soft Thresholding Functions  #####

def soft_2d_gradient2_rgb(model, v,h,tau):
    

    z0 = torch.tensor(0, dtype = torch.float32, device=model.cuda_device)
    z1 = torch.zeros(model.batch_size, 3, 1, model.DIMS1*2, dtype = torch.float32, device=model.cuda_device)
    z2 = torch.zeros(model.batch_size, 3, model.DIMS0*2, 1, dtype= torch.float32, device=model.cuda_device)

    vv = torch.cat([v, z1] , 2)
    hh = torch.cat([h, z2] , 3)
    
    #adding some small value so to solve the non gradient 
    mag = torch.sqrt(vv*vv + hh*hh+torch.tensor(1.11e-14))
    magt = torch.max(mag - tau, z0, out=None)
    mag = torch.max(mag - tau, z0, out=None) + tau
    mmult = magt/(mag)#+1e-5)

    return v*mmult[:,:, :-1,:], h*mmult[:,:, :,:-1]

# computes the HtX
def Hadj(model,x):
    xc = torch.zeros_like(x, dtype=torch.float32)
    x_complex=torch.complex(x, xc)
    X = torch.fft.fft2(x_complex)
    Hconj=model.Hconj_new
    
    HX = Hconj*X
    out = torch.fft.ifft2(HX)
    out_r=out.real
    return out_r

#computes the uk+1
def Ltv_tf(a, b): 
    return torch.cat([a[:,:, 0:1,:], a[:,:, 1:, :]-a[:,:, :-1, :], -a[:,:,-1:,:]],
                2) + torch.cat([b[:,:,:,0:1], b[:, :, :, 1:]-b[:, :, :,  :-1], -b[:,:, :,-1:]],3)
    
#takes the real matrix and return the corresponplex matrix
def make_complex(r, i = 0):
    if i==0:
        i = torch.zeros_like(r, dtype=torch.float32)
    return torch.complex(r, i) 

#computes the Hx+1
def Hfor(model, x):
    xc = torch.zeros_like(x, dtype=torch.float32)
    x_complex=torch.complex(x, xc)
    #print(x.shape)
    X = torch.fft.fft2(x_complex)
    HX = model.H*X
    out = torch.fft.ifft2(HX)
    return out.real

######## ADMM Parameter Update #########
def param_update(mu, res_tol, mu_inc, mu_dec, r, s):
    
    if r > res_tol * s:
        mu_up = mu*mu_inc
    else:
        mu_up = mu
        
    if s > res_tol*r:
        mu_up = mu_up/mu_dec
    else:
        mu_up = mu_up

def crop(model, x):
    C01 = model.PAD_SIZE0; C02 = model.PAD_SIZE0 + model.DIMS0              # Crop indices 
    C11 = model.PAD_SIZE1; C12 = model.PAD_SIZE1 + model.DIMS1              # Crop indices 
    return x[:, :, C01:C02, C11:C12]

def TVnorm_tf(x):
    x_diff, y_diff = L_tf(x)
    result = torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))
    return result

######## normalize image #########
def normalize_image(image):
    out_shape = image.shape
    image_flat = image.reshape((out_shape[0],out_shape[1]*out_shape[2]*out_shape[3]))
    image_max,_ = torch.max(image_flat,1)
    image_max_eye = torch.eye(out_shape[0], dtype = torch.float32, device=image.device)*1/image_max
    image_normalized = torch.reshape(torch.matmul(image_max_eye, image_flat), (out_shape[0],out_shape[1],out_shape[2],out_shape[3]))
    
    return image_normalized
