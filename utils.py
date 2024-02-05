import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

def preplot(image):
    #convert from CHW to HWC for matplotlib
    # #convert numpy array to torch tensor
    image=torch.tensor(image)
    #change CHW to HWC for matplotlib
    image=torch.permute(image,(1,2,0))
    #change BGR to RGB
    image=image[:,:,[2,1,0]]
    #limit the value to 0-1
    out=torch.clip(image,0,1)
    # flip the image
    out=torch.flipud(out)
    #return the cropped image
    return out[60:,62:-38,:]


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def downsample_ax2(img, factor):
    n = int(np.log2(factor))
    for i in range(n):
        if len(img.shape) == 2:
            img = .25 * (img[::2, ::2] + img[1::2, ::2]
                + img[::2, 1::2] + img[1::2, 1::2])
        else:
            img = .25 * (img[::2, ::2, :] + img[1::2, ::2, :]
                + img[::2, 1::2, :] + img[1::2, 1::2, :])
    return(img)

def load_psf_image2(psf_file, downsample=400, rgb=True):

    #converts psf to greyscale
    if rgb==True:
        my_psf = rgb2gray(np.array(Image.open(psf_file)))
    else:
        my_psf = np.array(Image.open(psf_file))
        
    psf_bg = np.mean(my_psf[0 : 15, 0 : 15])             #102
    psf_down = downsample_ax2(my_psf - psf_bg, downsample)
    
    psf_down = psf_down/np.linalg.norm(psf_down)
    
    return(psf_down)
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", experiment_name,timestamp, model_name, extra)
    else:
        log_dir = os.path.join("runs", experiment_name,timestamp, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

#Exponential Moving Average
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())