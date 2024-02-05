
import os
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

NUM_WORKERS = 3

class CustomDataFolder(Dataset):
    """
    Diffuser cam dataset Dataloader.
    """
    
    def __init__(self,csv_file,data_dir, label_dir,transform,ds=None):
        
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the Diffuser images.
            label_dir (string): Directory with all the natural images.
            ds: downsampling of image
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.csv_contents= pd.read_csv(csv_file,header=None)
        self.data_dir= data_dir
        self.label_dir=label_dir
        self.transform=transform
        self.ds=ds
        
        
    def __len__(self):
            return len(self.csv_contents)
        
        
    def __getitem__(self,index):
            
            image_name=self.csv_contents.iloc[index,0]
            
            path_diffuser=os.path.join(self.data_dir,image_name)
            path_gt=os.path.join(self.label_dir,image_name)
            
            image=np.load(path_diffuser[:-9]+'.npy')
            label=np.load(path_gt[0:-9]+'.npy')
            
            sample = {'image': torch.from_numpy(image.transpose(2,0,1)), 'label': torch.from_numpy(label.transpose(2,0,1))}

            if self.transform:
                
                sample = {'image':self.transform(sample['image']), 'label':self.transform(sample['label'])}
            return sample
          
def create_dataloaders(
    train_csv:str,
    test_csv:str,
    data_dir: str, 
    label_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.
  """
  # Use ImageFolder to create dataset(s)
  train_data=CustomDataFolder(train_csv,data_dir,label_dir,transform)
  test_data=CustomDataFolder(test_csv,data_dir,label_dir,transform)


  #Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader
