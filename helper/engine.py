"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from lpips import LPIPS
from utils import *
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import numpy as np
from helper.logging_metric import LogMetric
import torch.nn as nn
import pickle
import os
import copy
from utils import create_writer
from utils import EMA
import warnings
warnings.filterwarnings('ignore')
#setup EMA
ema = EMA(0.995)




def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,epoch:int,num_epochs:int,ema_model):
    """Trains a PyTorch model for a single epoch.
    """
    # Put model in train mode
    model.train()
    #mse loss 
    mse_loss = torch.nn.MSELoss()

    # Setup train loss 
    total_loss_org = 0
    train_batches = 0

    tqdm_loader = tqdm(dataloader, desc =f'Training Epoch:{epoch+1}')
     
    # Loop through data loader data batches
    for batch, data in enumerate(tqdm_loader):
        # Send data to target device
        images=data['image']
        labels=data['label']
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        if not(torch.any(output == float('-inf'))):
            optimizer.zero_grad()
            loss_normal=mse_loss(output,labels)
            
            #LPIPS
            #loss_normal = lpips_loss(output, labels)
            #  #weighted MSE and LPIPS
            # loss_mse = mse_loss(output, labels)
            # loss_lpips = lpips_loss(output, labels)
            # #Set the weights for MSE and LPIPS losses - linear switch over epochs
            # mse_weight = 1 - np.maximum((epoch-20),0) / num_epochs
            # lpips_weight = np.maximum(epoch-20,0) / num_epochs
            # loss_normal = mse_weight * loss_mse + lpips_weight * loss_lpips
            
            # skip the batch is loss is nan
            # Check if loss is NaN
            # if torch.isnan(loss_normal).any():
            #     print(loss_normal)
            #     print('error here')
            #     continue
            #loss_normal=mse_loss(output,labels)
            loss_normal.backward()
            
            #zero out NAN gradients
            zero_out_nan_gradients(model)
            optimizer.step()
            ema.step_ema(ema_model,model)
            
            total_loss_org += loss_normal.data.item()
            train_batches += 1
            tqdm_loader.set_postfix({
            'epoch': epoch+1, 
            'batch': batch+1, 
            'total_batches': len(tqdm_loader), 
            'avg_loss': total_loss_org /(batch + 1)
            })
        else:
            print('skipping batch')
    # Adjust metrics to get average loss 
    total_loss_org /= train_batches
    print("Traing loss:",total_loss_org)
    return total_loss_org

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,lpipsloss, 
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.
    """
    # Put model in eval mode
    model.eval() 
    

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
        for batch, data in enumerate(dataloader):
            print('\r', 'running test images, image:', batch, end = '')
            # Send data to target device
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            # 1. Forward pass
            output= model(inputs)

            if not(np.any(output.cpu().detach().numpy() == -np.inf)):

                mse_batch = mse_loss(output, labels)
                lpips_batch = lpipsloss.forward(output.cpu().detach(), labels.cpu().detach())

            

                #lpips_center = lpipsloss.forward(output[:, c1-sz:c1+sz, c2-sz:c2+sz], inputs[:, c1-sz:c1+sz, c2-sz:c2+sz])
                psnr_batch = 20 * torch.log10(1 / torch.sqrt(mse_batch))

                loss_dict['mse'].append(mse_batch.cpu().detach().numpy().squeeze())
                loss_dict['lpips'].append(lpips_batch.cpu().detach().numpy().squeeze())

                #loss_dict['lpips_center'].append(lpips_center.cpu().detach().numpy().squeeze())
                #loss_dict['lpips'].append(lpips_batch)
                loss_dict['psnr'].append(psnr_batch.cpu().detach().numpy().squeeze())

                # if batch == 63:
                #     loss_dict['sample_image'] = preplot(output.detach().cpu().numpy()[0])

        loss_dict['mse_avg'] = np.average(loss_dict['mse'][:-1]).squeeze()
        loss_dict['psnr_avg'] = np.average(loss_dict['psnr'][:-1]).squeeze()
        loss_dict['lpips_avg'] = np.average(loss_dict['lpips'][:-1]).squeeze()
        #loss_dict['lpips_center_avg'] = np.average(loss_dict['lpips_center']).squeeze()
        print('\n', 'avg mse:', loss_dict['mse_avg'], 'avg psnr:', 
              loss_dict['psnr_avg'], 'avg lpips:', loss_dict['lpips_avg'])

        return  loss_dict['mse_avg'],loss_dict['psnr_avg'],loss_dict['lpips_avg']
        
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          val=True,
          checkpoint_path: str = None,
          ) -> Dict[str, List]:
    """Trains and tests a PyTorch model."""

    # initialize the logging dictionary
    #metric_dict = LogMetric( { 'train_loss' : [],'epoch':[]})
      # Create empty results dictionary
    results = {"train_loss": [],
               "eval_mse": [],
               "eval_psnr": [],
               "eval_lpips": []
    }
    # Make sure model on target device
    model.to(device)
    #lpips to device
    loss_fn_vgg = LPIPS(net='alex')
    #lpips loss
    lpipsloss = loss_fn_vgg.to('cpu')
    
    
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    #print(ema_model)


    # Load checkpoint if exists
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        results = checkpoint['metric_dict']
        start_epoch = checkpoint['epoch'] + 1
        print("Loaded checkpoint and continuing training from epoch", start_epoch)

    # Loop through training and testing steps for a number of epochs
    for epoch in range(start_epoch, epochs):
        #print(epoch,epochs)
        train_loss = train_step(model=model,
                                        dataloader=train_dataloader,
                                        optimizer=optimizer,
                                        device=device,epoch=epoch,num_epochs=epochs,ema_model=ema_model)
        
        if val==True:
            eval_mse,eval_psnr,eval_lpips= test_step(model=model,
                dataloader=test_dataloader,lpipsloss=lpipsloss,
                device=device)

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["eval_mse"].append(eval_mse)
        results["eval_psnr"].append(eval_psnr)
        results["eval_lpips"].append(eval_lpips)
            # Update results dictionary
        '''adding to the dictionary'''
        
        #add writer
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Train MSE", 
                            tag_scalar_dict={"train_loss": train_loss,
                                                },
                            global_step=epoch)
            writer.add_scalars(main_tag="Validation ", 
                            tag_scalar_dict={"Eval_MSE": eval_mse,
                                                "Eval_PSNR": eval_psnr,
                                                "Eval_LPIPS": eval_lpips}, 
                            global_step=epoch)

            # Close the writer
            writer.close()
            
            #metric_dict.update_dict([eval_mse,eval_psnr,eval_lpips],training=False)
        if epoch % 1 == 0:
            save_checkpoint(model, optimizer, results, epoch, f"/home/arpanp/diffuser_cam_vit/experiments_SWIN/checkpoints_unet_self/checkpoint-{epoch}.pth")
            save_checkpoint(ema_model, None, None, None, f"/home/arpanp/diffuser_cam_vit/experiments_SWIN/checkpoints_unet_self/checkpoint_ema-{epoch}.pth")
            


    with open('unet10_metrics_mse.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Return the filled results at the end of the epochs
    return results

def save_checkpoint(model, optimizer, results, epoch, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metric_dict': results
    }, filename)

def zero_out_nan_gradients(model):
    for name, p in model.named_parameters():
            if p.grad is not None:
                p.grad[p.grad!=p.grad]=0.