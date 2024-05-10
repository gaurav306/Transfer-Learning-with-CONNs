import datetime as dt
import os
from typing import *
import time

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.callbacks import Callback

os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'

class Timer():
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        timetaken = end_dt - self.start_dt
        print('Time taken: %s' % (timetaken))
        #print("Current memory (MBs)",get_memory_info('GPU:0')['current'] / 1048576)
        print("")
        return timetaken

class PrintLoggingCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        logs['epoch_duration'] = epoch_duration



def fill_nan(test_np): # fill nan with previous value # test_np should be 3D NumPy ndarray
    for i in range(test_np.shape[2]):
        arr = test_np[:,:,i]
        if np.isnan(arr).any():
            mask = np.isnan(arr)
            idx = np.where(~mask,np.arange(mask.shape[1]),0)
            np.maximum.accumulate(idx,axis=1, out=idx)
            out = arr[np.arange(idx.shape[0])[:,None], idx]
            test_np[:,:,i] = out

    return test_np


def average(lst: List[float]) -> float:
	"""Return the average of the list of numbers lst."""
	# Sum the numbers and divide by the number of numbers.
	return sum(lst) / len(lst)


class CustomEarlyStopping(Callback):
    def __init__(self, configs, ident, check_epoch_index, min_mape, error_type, **kwargs):
        super(CustomEarlyStopping, self).__init__(**kwargs)
        self.zero_epoch_mape = None
        self.check_epoch_mape = None
        self.check2_epoch_mape = None
        self.check_epoch_index = check_epoch_index
        self.check2_epoch_index = 10
        
        self.min_mape = min_mape
        self.error_type = error_type
        self.ident = ident
        self.configs = configs

    def on_epoch_end(self, epoch, logs=None):
        current_mape = logs.get(self.error_type)
        if epoch == 0:
            print("MAPE at 1st epoch is being recorded by CustomEarlyStopping: ", current_mape)
            self.zero_epoch_mape = current_mape
            if self.zero_epoch_mape > self.min_mape:
                self.model.stop_training = True
                print("Stopping training: MAPE at 1st epoch is greater than %s" % self.min_mape)
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")
                raise SystemExit("Early Stopping Condition met - Restarting Training \n")
        elif epoch == self.check_epoch_index:
            print("MAPE at check epoch is being recorded by CustomEarlyStopping: ", current_mape)
            self.check_epoch_mape = current_mape
        elif epoch == self.check2_epoch_index:
            self.check2_epoch_mape = current_mape
            
        if self.zero_epoch_mape and self.check_epoch_mape:
            if self.check_epoch_mape > self.zero_epoch_mape / 2:
                self.model.stop_training = True
                print("Stopping training: MAPE at check epoch is more than half of MAPE at 1st epoch")
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")
                raise SystemExit("Early Stopping Condition met - Restarting Training \n")
        if self.check_epoch_mape and self.check2_epoch_mape:
            if self.check2_epoch_mape > self.check_epoch_mape:
                self.model.stop_training = True
                print("Stopping training: MAPE at check epoch is more than MAPE at check2 epoch")
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")
                raise SystemExit("Early Stopping Condition met - Restarting Training \n")
        """
        with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
            file.write("False")
        """

class CustomEarlyStopping_tuning(Callback):
    def __init__(self, configs, ident, check_epoch_index, min_mape, error_type, **kwargs):
        super(CustomEarlyStopping_tuning, self).__init__(**kwargs)
        self.zero_epoch_mape = None
        self.check_epoch_mape = None
        self.check2_epoch_mape = None
        self.check_epoch_index = check_epoch_index
        self.check2_epoch_index = 3
        
        self.min_mape = min_mape
        self.error_type = error_type
        self.ident = ident
        self.configs = configs

    def on_epoch_end(self, epoch, logs=None):
        current_mape = logs.get(self.error_type)
        if epoch == 0:
            print("MAPE at 1st epoch is being recorded by CustomEarlyStopping: ", current_mape)
            self.zero_epoch_mape = current_mape
            if self.zero_epoch_mape > self.min_mape:
                self.model.stop_training = True
                print("Stopping training: MAPE at 1st epoch is greater than %s" % self.min_mape)
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")
                raise SystemExit("Early Stopping Condition met - Restarting Training \n")
        elif epoch == self.check_epoch_index:
            print("MAPE at check epoch is being recorded by CustomEarlyStopping: ", current_mape)
            self.check_epoch_mape = current_mape
        elif epoch == self.check2_epoch_index:
            print("MAPE at check epoch is being recorded by CustomEarlyStopping: ", current_mape)
            self.check2_epoch_mape = current_mape
            
        if self.zero_epoch_mape and self.check_epoch_mape:
            if self.check_epoch_mape > self.zero_epoch_mape :
                self.model.stop_training = True
                print("Stopping training: MAPE at check epoch is more than MAPE at 1st epoch")
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")
                raise SystemExit("Early Stopping Condition met - Restarting Training \n")
        if self.check_epoch_mape and self.check2_epoch_mape:
            if round(self.check2_epoch_mape,5) == round(self.check_epoch_mape,5):
                self.model.stop_training = True
                print("Stopping training: MAPE at check epoch is equal to MAPE at check2 epoch")
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")
                raise SystemExit("Early Stopping Condition met - Restarting Training \n")
        """
        with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
            file.write("False")
        """

from tensorflow.keras.callbacks import TerminateOnNaN

class CustomTerminateOnNaN(TerminateOnNaN):
    def __init__(self, configs, ident):
        super(CustomTerminateOnNaN, self).__init__()
        self.ident = ident
        self.configs = configs

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print(f'NaN or inf loss encountered at {batch}, terminating training and flagging for restart...')
                self.model.stop_training = True
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")


import datetime as dt
import os
from typing import *
import time

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.callbacks import Callback

class SGDRScheduler_warmup(Callback):       #modified to have warmup every restart
    '''Cosine annealing learning rate scheduler with periodic warmup restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=30,
                                     cycle_mult_factor=2,
                                     warmup_length=10,
                                     warmup_mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        cycle_mult_factor: Scale cycle_length after each full cycle completion.
        warmup_length: Number of epochs for warmup during every restart. #minimum 1
        warmup_mult_factor: Scale warmup_length after each full cycle completion.
        
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
        with Warm Restarts. https://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay,
                 cycle_length,
                 cycle_mult_factor,
                 warmup_length,
                 warmup_mult_factor):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.lr_decay = lr_decay
        self.cycle_length = cycle_length
        self.cycle_mult_factor = cycle_mult_factor
        if warmup_length == 0:
            warmup_length = 1
        self.warmup_length = warmup_length
        self.warmup_mult_factor = warmup_mult_factor
        
        self.batch_since_restart = 0
        self.batch_since_warmup = 0
        self.warmup_cycle_mode = 0  #warmup =0, cycle=1
        self.next_restart = warmup_length #we start with warmup
        self.flag = 0

        self.history = {}
        print('[TrainClass] CWR added as callback')

    def clr(self):
        '''Calculate the learning rate cosine fall.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr
    
    def warmup_clr(self):
        '''Calculate the learning rate during warmup.'''
        fraction_to_restart = self.batch_since_warmup / (self.steps_per_epoch * self.warmup_length)
        #lr = self.min_lr + (self.max_lr - self.min_lr) * fraction_to_restart
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 - np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        #K.set_value(self.model.optimizer.lr, self.max_lr) # when no warmup
        K.set_value(self.model.optimizer.lr, self.min_lr) # when warmup we start from lowest lr

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        if self.warmup_cycle_mode==0:
            K.set_value(self.model.optimizer.lr, self.warmup_clr()) 
            self.batch_since_warmup += 1     
        if self.warmup_cycle_mode==1:
            K.set_value(self.model.optimizer.lr, self.clr())
            self.batch_since_restart += 1
            

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.warmup_cycle_mode = not self.warmup_cycle_mode # swap 0 to 1 and vice versa
            
            self.batch_since_restart = 0
            self.batch_since_warmup = 0
            
            if self.warmup_cycle_mode==0: 
                self.warmup_length = np.ceil(self.warmup_length * self.warmup_mult_factor)
                self.next_restart += self.warmup_length
            if self.warmup_cycle_mode==1: 
                self.cycle_length = np.ceil(self.cycle_length * self.cycle_mult_factor)
                self.next_restart += self.cycle_length
            self.flag = self.flag + 1
            if self.flag == 1:
                self.warmup_length = 1           
            if self.flag == 2:
                self.max_lr = self.max_lr * 0.5
                self.warmup_length = 1
            else:
                self.max_lr *= self.lr_decay
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)  


#both these are used in custom training loop
def cyclic_lr(batch_since_restart, cycle_length, min_lr, max_lr):
    '''Calculate the learning rate cosine fall.'''
    fraction_to_restart = batch_since_restart / cycle_length
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
    return lr

def warmup_clr(batch_since_warmup, warmup_length, min_lr, max_lr):
    '''Calculate the learning rate during warmup.'''
    fraction_to_restart = batch_since_warmup / warmup_length
    lr = min_lr + (max_lr - min_lr) * fraction_to_restart
    return lr



def mean_bias_error(y_true, y_pred):
    return np.mean(y_true - y_pred)

def normalized_mean_bias_error(y_true, y_pred):
    return mean_bias_error(y_true, y_pred) / np.mean(y_true, axis=0) * 100

def coefficient_of_variation_root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred), axis=0)) / np.mean(y_true, axis=0) * 100

def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred), axis=0))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=0)

def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred), axis=0)

def r_square(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.var(y_true) * len(y_true)
    return 1 - ss_res / ss_tot

'''
from dtw import *
def dtw_distance1(y_true, y_pred):
    alignment = dtw(y_true, y_pred, keep_internals=True)
    return alignment.normalizedDistance
'''
from fastdtw import fastdtw
def dtw_distance(y_true, y_pred):
    distance, path = fastdtw(y_true, y_pred, radius=50)
    return distance/len(y_true)

def dtw_distance2(y_true, y_pred):
    return 1