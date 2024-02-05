import numpy as np 
import torch
from models.admm_helper import *

def admm(model, in_vars, alpha2k_1, alpha2k_2, CtC, Cty, mu_auto, n, y):
    """
    performs one iteration of ADMM
    """    
    
    sk = in_vars[0]
    alpha1k = in_vars[1] 
    alpha3k = in_vars[2]
    Hskp = in_vars[3]; 
    
    #if the autotune is enabled mu are from the mu_auto parameter
    if model.autotune == True:
        mu1 = mu_auto[0]
        mu2 = mu_auto[1]
        mu3 = mu_auto[2]
    #else mu are n-1 values of the mu list
    else:
        mu1 = model.mu1[n]
        mu2 = model.mu2[n]
        mu3 = model.mu3[n]
    # tau is the n-1 values of the iterations 
    tau = model.tau[n] #model.mu_vals[3][n]
    
    dual_resid_s = []
    primal_resid_s = []
    dual_resid_u = []
    primal_resid_u = []
    dual_resid_w = []
    primal_resid_w = []
    cost = []
    #print(mu1.device)

    Smult = 1/(mu1.to(model.cuda_device)*model.HtH.to(model.cuda_device) + mu2.to(model.cuda_device)*model.LtL.to(model.cuda_device) + mu3.to(model.cuda_device))  # May need to expand dimensions 
    Vmult = 1/(CtC + mu1)
    
    ###############  update u = soft(Ψ*x + η/μ2,  tau/μ2) ###################################
    Lsk1, Lsk2 = L_tf(sk)        # X and Y Image gradients 
    ukp_1, ukp_2 = soft_2d_gradient2_rgb(model, Lsk1 + alpha2k_1/mu2, Lsk2 + alpha2k_2/mu2, tau)
    
    ################  update      ######################################

    vkp = Vmult*(mu1*(alpha1k/mu1 + Hskp) + Cty)
    
    ################  update w <-- max(alpha3/mu3 + sk, 0) ######################################


    zero_cuda = torch.tensor(0, dtype = torch.float32, device=model.cuda_device)
        
    wkp = torch.max(alpha3k/mu3 + sk, zero_cuda, out=None)
   

    
    # no learned prox 
    skp_numerator = mu3*(wkp - alpha3k/mu3) + mu1 * Hadj(model, vkp - alpha1k/mu1) + mu2*Ltv_tf(ukp_1 - alpha2k_1/mu2, ukp_2 - alpha2k_2/mu2) 
    symm = []
  
    #SKP_numerator = torch.fft.fft(make_complex(skp_numerator), 2)
    SKP_numerator = torch.fft.fft2(make_complex(skp_numerator))

    skp = (torch.fft.ifft2((make_complex(Smult)* SKP_numerator))).real
    
    Hskp_up = Hfor(model, skp)
    r_sv = Hskp_up - vkp
    dual_resid_s.append(mu1 * torch.norm(Hskp - Hskp_up))
    primal_resid_s.append(torch.norm(r_sv))

    # Autotune
    if model.autotune == True:
        mu1_up = param_update(mu1, model.resid_tol, model.mu_inc, model.mu_dec, primal_resid_s[-1], dual_resid_s[-1])
        #model.mu_vals[0][n+1] = model.mu_vals[0][n+1] + mu1_up
    else: 
        if n == model.iterations-1:
            mu1_up = model.mu_vals[0][n]
        else:
            mu1_up = model.mu_vals[0][n+1]

    alpha1kup = alpha1k + mu1*r_sv

    Lskp1, Lskp2 = L_tf(skp)
    r_su_1 = Lskp1 - ukp_1
    r_su_2 = Lskp2 - ukp_2

    dual_resid_u.append(mu2*torch.sqrt(torch.norm(Lsk1 - Lskp1)**2 + torch.norm(Lsk2 - Lskp2)**2))
    primal_resid_u.append(torch.sqrt(torch.norm(r_su_1)**2 + torch.norm(r_su_2)**2))

    if model.autotune == True:
        mu2_up = param_update(mu2, model.resid_tol, model.mu_inc, model.mu_dec, primal_resid_u[-1], dual_resid_u[-1])
    else:
        if n == model.iterations-1:
            mu2_up = model.mu_vals[1][n]
        else:
            mu2_up = model.mu_vals[1][n+1]

    alpha2k_1up= alpha2k_1 + mu2*r_su_1
    alpha2k_2up= alpha2k_2 + mu2*r_su_2

    r_sw = skp - wkp
    dual_resid_w.append(mu3*torch.norm(sk - skp))
    primal_resid_w.append(torch.norm(r_sw))

    if model.autotune == True:
        mu3_up = param_update(mu3, model.resid_tol, model.mu_inc, model.mu_dec, primal_resid_w[-1], dual_resid_w[-1])
    else:
        if n == model.iterations-1:
            mu3_up = model.mu_vals[2][n]
        else:
            mu3_up = model.mu_vals[2][n+1]

    alpha3kup = alpha3k + mu3*r_sw

    data_loss = torch.norm(crop(model, Hskp_up)-y)**2
    tv_loss = tau*TVnorm_tf(skp)

    
    if model.printstats == True:
        
        admmstats = {'dual_res_s': dual_resid_s[-1].cpu().detach().numpy(),
                     'primal_res_s':  primal_resid_s[-1].cpu().detach().numpy(),
                     'dual_res_w':dual_resid_w[-1].cpu().detach().numpy(),
                     'primal_res_w':primal_resid_w[-1].cpu().detach().numpy(),
                     'dual_res_u':dual_resid_s[-1].cpu().detach().numpy(),
                     'primal_res_u':primal_resid_s[-1].cpu().detach().numpy(),
                     'data_loss':data_loss.cpu().detach().numpy(),
                     'total_loss':(data_loss+tv_loss).cpu().detach().numpy()}
        
        
        print('\r',  'iter:', n,'s:', admmstats['dual_res_s'], admmstats['primal_res_s'], 
         'u:', admmstats['dual_res_u'], admmstats['primal_res_u'],
          'w:', admmstats['dual_res_w'], admmstats['primal_res_w'], end='')
    else:
        admmstats = []

    
    #out vars contains X, alpha1k+1, alpha3k+1, Hxk+1
    out_vars = torch.stack([skp, alpha1kup, alpha3kup, Hskp_up])

    #updated value of mues
    mu_auto_up = torch.stack([mu1_up, mu2_up, mu3_up])
    # returns outvars, alpha2's, myus, admmstats
    return out_vars, alpha2k_1up, alpha2k_2up, mu_auto_up, symm, admmstats
    
    
