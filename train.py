
import os
import sys
import torch
from torchvision import transforms
from helper import data_setup, engine, model_builder
import  utils
import argparse
from timeit import default_timer as timer
from models.admm_model import *
from models.network_swinir import SwinIR as net
from models.ensemble import EnsembleModel
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
from models.u_net2 import Unet as unet2
from utils import create_writer
warnings.filterwarnings('ignore')
#torch.autograd.set_detect_anomaly(True)
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def get_args_parser():
    parser = argparse.ArgumentParser('Set params for the training', add_help=False)
    
    parser.add_argument('--batch', default=2, type=int, help=' number of batches')
    parser.add_argument('--epoch', default=50, type=int, help='number of epochs to train')
    parser.add_argument('--lr', default= 1e-4, type=int, help='number of epochs to train')
    parser.add_argument('--iter', default= 5, type=int, help='number of iterations to unroll')
    return parser

def main(args):
    
    # Setup hyperparameters
    NUM_EPOCHS = args.epoch
    BATCH_SIZE = args.batch
    LEARNING_RATE = args.lr
    iterations=args.iter

    # Setup directories
    path = str(os.getcwd()) 
    train_csv= path + '\\dataset\\dataset_train.csv'
    test_csv=path + '\\dataset\\dataset_test.csv'
    data_dir=path + '\\dataset\\diffuser_images\\'
    label_dir= path +'\\dataset\\ground_truth_lensed\\'

    # Setup target device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    #pad the image so that the image is compatible to swinIR
    train_transform=None
    # Create DataLoaders with help from data_setup.py
    train_dataloader,test_dataloader=data_setup.create_dataloaders(
                                    train_csv=train_csv,
                                    test_csv=test_csv,
                                    data_dir=data_dir,
                                    label_dir=label_dir,
                                    transform= train_transform,
                                    batch_size=BATCH_SIZE 
)

    # Create model with help from model_builder.py
    
    model_admm = model_builder.build_model(batch_size=BATCH_SIZE,iterations=iterations, device = device)
    '''model_swin=net(upscale=1, in_chans=3, img_size=(272,488), window_size=8,
                    img_range=1., depths=[1,1,1], embed_dim=180, num_heads=[6, 6,6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
                    '''
            
    
    model_unet=unet2(dim=36,channels=3,dim_mults=(1,2,4,8)) 
    #define ensembled model
    model=EnsembleModel(model_admm,model_unet)
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=LEARNING_RATE)
    
    # start time
    start_time=timer()
    # Start training with help from engine.py
    result=engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 writer=create_writer(experiment_name='unet_atten',model_name='unet_atten',extra=f"{NUM_EPOCHS}_epochs"),
                 device=device,val=True,checkpoint_path=None)
    # end time
    end_time=timer()
    print(f"Time to train {end_time-start_time:.3f}")
    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="/home/arpanp/diffuser_cam_vit/experiments_SWIN/checkpoints_unet_self",
                     model_name="unet_self.pth")
    print(result)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser('training script', parents=[get_args_parser()])
    #print(os.getcwd())
    args = parser.parse_args()
    main(args)  
    