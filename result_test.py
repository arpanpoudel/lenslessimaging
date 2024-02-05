from helper import model_builder
import torch
from models.network_swinir import SwinIR as net
from models.ensemble import MyEnsemble
from models.u_net2 import Unet as unet
from models import unet as unet_model
from helper import data_setup
import os
from lpips import LPIPS
from tqdm import tqdm
from timeit import default_timer as timer
import numpy as np
import models.admm_model as admm_model_plain
# Initializing the LPIPS model
import time

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              device: torch.device):
    """Tests a PyTorch model for a single epoch.
    """
    model.eval() 
    
    #lpips loss
    lpipsloss = loss_fn_vgg.to('cpu')
    #mse loss 
    mse_loss = torch.nn.MSELoss()
    #setup loss dict
    loss_dict = {'mse': [], 'mse_avg': 0, 
                 'psnr':[], 'psnr_avg': 0,
                 'lpips': [], 'lpips_avg':0,
                }
    #

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        tqdm_loader = tqdm(dataloader, desc =f'Testing')
        for batch, data in enumerate(tqdm_loader):
            #print('\r', 'running test images, image:', batch, end = '')
            # Send data to target device
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            s=timer()
            # 1. Forward pass
            output= model(inputs)
            e=timer()
            tt=e-s
            if not(np.any(output.cpu().detach().numpy() == -np.inf)):

                mse_batch = mse_loss(output, labels)
                lpips_batch = lpipsloss.forward(output.cpu().detach(), labels.cpu().detach())
                psnr_batch = 20 * torch.log10(1 / torch.sqrt(mse_batch))

                loss_dict['mse'].append(mse_batch.cpu().detach().numpy().squeeze())
                loss_dict['lpips'].append(lpips_batch.cpu().detach().numpy().squeeze())

                loss_dict['psnr'].append(psnr_batch.cpu().detach().numpy().squeeze())
            tqdm_loader.set_postfix({
                        'batch': batch+1,
                        'time':tt,
                        'mse': loss_dict['mse'][-1],
                        })
        loss_dict['mse_avg'] = np.average(loss_dict['mse'][:-1]).squeeze()
        loss_dict['psnr_avg'] = np.average(loss_dict['psnr'][:-1]).squeeze()
        loss_dict['lpips_avg'] = np.average(loss_dict['lpips'][:-1]).squeeze()
        print('\n', 'avg mse:', loss_dict['mse_avg'], 'avg psnr:', 
              loss_dict['psnr_avg'], 'avg lpips:', loss_dict['lpips_avg'])

        return  loss_dict['mse_avg'],loss_dict['psnr_avg'],loss_dict['lpips_avg']
    
    
loss_fn_vgg = LPIPS(net='alex')

def lpips_loss(output, target):
    return loss_fn_vgg.forward(output, target).mean()

def run_timing_test(model, test_loader, device, num_trials = 100):
    print('\r', 'running timing test', end = '')
    t_avg_gpu = 0
    t_avg_cpu = 0
    
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            inputs = sample_batched['image'].to(device); 
            break

    print('\r', 'running GPU timing test', end = '')
    for i in range(0,num_trials):
        with torch.no_grad():
            t = time.time()
            output = model(inputs)
            elapsed = time.time() - t
            t_avg_gpu = t_avg_gpu + elapsed


    t_avg_gpu = t_avg_gpu/num_trials 
    
    return t_avg_gpu

def main():
    BATCH_SIZE=2
    iterations=5
    device='cuda:6'
    path = str(os.getcwd()) 
    train_csv= path + '/dataset/dataset/dataset_train.csv'
    test_csv=path + '/dataset/dataset/dataset_test.csv'
    data_dir=path + '/dataset/dataset/diffuser_images/'
    label_dir= path +'/dataset/dataset/ground_truth_lensed/'
    train_dataloader,test_dataloader=data_setup.create_dataloaders(
                                    train_csv=train_csv,
                                    test_csv=test_csv,
                                    data_dir=data_dir,
                                    label_dir=label_dir,
                                    transform= None,
                                    batch_size=BATCH_SIZE)
    
    model_admm = model_builder.build_model(batch_size=BATCH_SIZE,iterations=iterations, device = device)   
    admm = model_builder.build_model(batch_size=BATCH_SIZE,iterations=iterations, device = device)
    model_unet=unet(dim=36,channels=3,dim_mults=(1,2,4,8)).to(device)
    model=MyEnsemble(admm,model_unet)
    checkpoint=torch.load('/home/arpanp/diffuser_cam_vit/experiments_SWIN/checkpoints_unet_self/checkpoint-49.pth',map_location=device)
    model.load_state_dict(checkpoint)
    start_time = timer()
    result_test = run_timing_test(model, test_dataloader,device,num_trials=100)
    end_time = timer()
    print(result_test)
    print(f"\n Time to train {end_time-start_time:.3f}")
    
if __name__ == "__main__":
    main()