import torch.nn.functional as F
import numpy as np 
import torch

def pad_zeros_torch(model, x):
    PADDING = (model.PAD_SIZE1, model.PAD_SIZE1, model.PAD_SIZE0, model.PAD_SIZE0)
    return F.pad(x, PADDING, 'constant', 0)

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

#makes the laplacian filer used as the sparsyfying tranform
def make_laplacian(model):
    lapl = np.zeros([model.DIMS0*2,model.DIMS1*2])
    lapl[0,0] =4.; 
    lapl[0,1] = -1.; lapl[1,0] = -1.; 
    lapl[0,-1] = -1.; lapl[-1,0] = -1.; 

    LTL = np.abs(np.fft.fft2(lapl))
    # LTL = np.abs(np.fft.fft2(lapl))
    return LTL
