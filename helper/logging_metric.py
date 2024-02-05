import pickle
import os

class LogMetric(object):
    def __init__(self,training_dict=None):
        if training_dict:
            self.training_dict = training_dict
        else:
            self.training_dict = { 'train_loss' : []}
        self.val_dict={'val_mse': [],'val_psnr':[],'val_lpips':[]}
        self.log_dict ={}

    def update_dict_with_key(self,key,value,training=True):
        if training:
            self.training_dict[key].append(value)
        else:
            self.val_dict[key].append(value)

    def update_dict(self,value=[],training=True):
        for (k, v) in zip(self.get_dict_keys(training), value):
            if training:
                self.training_dict[k].append(v)
            else:
                self.val_dict[k].append(v)

    def get_dict_keys(self,training=True):
        if training:
            return list(self.training_dict.keys()) 
        else:
            return list(self.val_dict.keys()) 


    def update_log_dict(self):
        self.log_dict = {**self.training_dict, **self.val_dict}
        return self.log_dict

    def save_dict(self,opt,save_name='loss_metric.pkl'):
        if not os.path.exists(opt.loss_dir):
            os.makedirs(opt.loss_dir)
        save_path=os.path.join(opt.loss_dir,save_name)
        
        self.update_log_dict()
        with open(save_path,"wb") as fp:
            pickle.dump(self.log_dict,fp)
        return save_path 