__author__ = "Gaurav Chaudhary"
__copyright__ = "Gaurav Chaudhary 2022"
__version__ = "1.0.0"
__license__ = "MIT"

import os

import datetime as dt
import sys, glob
from src.configs.configs_init import merge_configs
from src.data_processor import DataClass
from src.main_run import Main_Run_Class
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def get_keywords_files(dir_path, keywords, ext):
    matching_files = []
    all_h5_files = glob.glob(dir_path + '*.'+ext, recursive=True)
    for file_path in all_h5_files:
        file_name = os.path.basename(file_path)
        if all(keyword in file_name for keyword in keywords):
            matching_files.append(file_path)
            print('Found file: {}'.format(file_name))

    if len(matching_files) > 1:
        raise ValueError('More than one file found for keywords: {}'.format(keywords))
    elif len(matching_files) == 0:
        print('No file found for keywords: {}'.format(keywords))
        return None
    else:
        return matching_files[0]




def test_validation_set_training4(configs, SAVEPATH, city, stoch, dt_ident, params):
    configs['if_model_image']           = 0
    configs['all_layers_neurons']       = 128
    configs['data']['input_EP_csv_files']   = ['TRD_OG_TL_nohvac_nowindow_ts4.csv', 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']
    configs['training']['batch_size']       = 512
    configs['data']['save_results_dir']     = SAVEPATH

    for week_num in range(0,27):
        ident = 'TRD_OG_tuned-CHECKED_run-%s_week-%s' % (stoch, week_num) + dt_ident
        TL_model_path = get_keywords_files(SAVEPATH, ['TL_model-TRD_OG_tuned-%s' % city, 'run-%s_%s' % (stoch, params), 'week-%s.' % (week_num)], 'h5')
        print(TL_model_path)
        run_pipeline = Main_Run_Class(ident, configs, TL_model_path)
        print('%s finished' % ident)

def test_validation_set_training41(configs, SAVEPATH, city, stoch, dt_ident, params):
    configs['if_model_image']           = 0
    configs['all_layers_neurons']       = 128
    configs['data']['input_EP_csv_files']   = ['TRD_OG_TL_nohvac_nowindow_ts4.csv', 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']
    configs['training']['batch_size']       = 512
    configs['data']['save_results_dir']     = SAVEPATH

    for week_num in range(0,6):
        ident = 'TRD_OG_tuned-CHECKED_run-%s_month-%s' % (stoch, week_num) + dt_ident
        TL_model_path = get_keywords_files(SAVEPATH, ['TL_model-TRD_OG_tuned-%s' % city, 'run-%s_%s' % (stoch, params), 'month-%s.' % (week_num)], 'h5')
        print(TL_model_path)
        run_pipeline = Main_Run_Class(ident, configs, TL_model_path)
        print('%s finished' % ident)

def test_validation_set_training42(configs, SAVEPATH, city, stoch, dt_ident, params):
    configs['if_model_image']           = 0
    configs['all_layers_neurons']       = 128
    configs['data']['input_EP_csv_files']   = ['TRD_OG_TL_nohvac_nowindow_ts4.csv', 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']
    configs['training']['batch_size']       = 512
    configs['data']['save_results_dir']     = SAVEPATH

    for week_num in range(0,26):
        ident = 'TRD_OG_tuned-CHECKED_run-%s_weekly-%s' % (stoch, week_num) + dt_ident
        TL_model_path = get_keywords_files(SAVEPATH, ['TL_model-TRD_OG_tuned-%s' % city, 'run-%s_%s' % (stoch, params), 'weekly-%s.' % (week_num)], 'h5')
        print(TL_model_path)
        run_pipeline = Main_Run_Class(ident, configs, TL_model_path)
        print('%s finished' % ident)



def og_case(ident, configs, runparams, model_path):
    batch, epoch, lr, lr_not, opt, city, stoch, runindex = runparams
    
    configs['if_model_image']           = 0
    configs['loss']                     = 6
    configs['all_layers_neurons']       = 128
    configs['training']['epochs']       = epoch
    configs['training']['EarlyStop_patience'] = 15
    configs['tl_case']['tl_type']      = 1
    configs['data']['input_EP_csv_files'] = ['M0_Steinkjer.csv', 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']

    configs['optimizer']                        = opt
    configs[str(configs['optimizer'])]['lr']    = lr
    configs['training']['batch_size']           = batch
    
    configs['data']['save_results_dir'] = 'saved_results/19july/OG_case1/'
    print('Current run case: ', ident)
    run_pipeline = Main_Run_Class(ident, configs, model_path)
    print('%s finished' % ident)


def multicase(ident, configs, runparams, model_path):
    batch, epoch, lr, lr_not, opt, city, stoch, runindex = runparams
    
    configs['if_model_image']           = 0
    configs['loss']                     = 6
    configs['all_layers_neurons']       = 128
    configs['training']['epochs']       = epoch
    configs['training']['EarlyStop_patience'] = 15
    configs['tl_case']['tl_type']      = 1
    configs['data']['input_EP_csv_files'] = ['%s_GEO_CH_TL_nohvac_nowindow_ts4.csv' % city, 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']

    configs['optimizer']                        = opt
    configs[str(configs['optimizer'])]['lr']    = lr
    configs['training']['batch_size']           = batch
    
    #configs['data']['save_results_dir'] = 'saved_results/19july/lr-%s_%s_%s_%s-%s/' % (str(lr_not), str(opt), str(batch), city, str(stoch))
    configs['data']['save_results_dir'] = 'saved_results/19july/lr-%s_%s_%s_%s-%s/' % (str(lr_not), str(opt), str(batch), city, str(runindex))
    print('Current run case: ', ident)
    run_pipeline = Main_Run_Class(ident, configs, model_path)
    print('%s finished' % ident)




if __name__ == '__main__':

    start_script = dt.datetime.now()
    data_config         = 'config_data.yaml'
    training_config     = 'config_training.yaml'
    
    runindex = int(sys.argv[1])
    RUNTHIS = int(sys.argv[2])

    city, i = 'BERGEN', '1'

    batch_sizes = [128] # [256, 512]
    epochs      = [75] # [25, 50]
    lrs         = [0.001, 0.0001, 0.00001, 0.000001] # 4
    lrs_not     = ['1e-3', '1e-4', '1e-5', '1e-6']
    optimizers  = ['SGD', 'Adam'] # 2

    allindexs = []
    for batch, epoch in zip(batch_sizes, epochs):
        for lr, lr_not in zip(lrs, lrs_not):
            for opt in optimizers:
                allindexs.append([batch, epoch, lr, lr_not, opt])
    
    #run_params = allindexs[runindex]
    #batch, epoch, lr, lr_not, opt = allindexs[runindex]
    
    run_params = [512, 25, 0.0001, '1e-4', 'Adam']
    batch, epoch, lr, lr_not, opt = run_params
    
    run_params.append(city)
    run_params.append(i)
    run_params.append(runindex)

    #dt_ident = '_lr-%s_%s_%s_%s-%s' % (str(lr_not), str(opt), str(batch), city, str(i)) + '_' + str(dt.datetime.now().strftime('%d.%H.%f')[:-2])
    dt_ident = '_lr-%s_%s_%s_%s-%s' % (str(lr_not), str(opt), str(batch), city, str(runindex)) + '_' + str(dt.datetime.now().strftime('%d.%H.%f')[:-2])


    if RUNTHIS == 1:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['TRD_OG_TL_BASE_ad_rMHA', 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 5
        ident = 'TL_model-TRD_OG_FULL_' + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        og_case(ident, configs_data, run_params, TL_model_path) 
        
        """
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['%s_GEO_CH_TL_FULL_ad_rMHA' % city, 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 5
        ident = 'TL_model-%s_GEO_CH_FULL_test-%s_run-%s' % (city, city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        multicase(ident, configs_data, run_params, TL_model_path)
        
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['%s_GEO_CH_TL_HALF_ad_rMHA' % city, 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 5
        ident = 'TL_model-%s_GEO_CH_HALF_test-%s_run-%s' % (city, city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        multicase(ident, configs_data, run_params, TL_model_path)
        """
    """
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['TRD_OG_TL_BASE_ad_rMHA', 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 4
        configs_data['tl_case']['start'] = 0
        configs_data['tl_case']['end']   = 27
        ident = 'TL_model-TRD_OG_tuned-%s_run-%s' % (city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        multicase(ident, configs_data, run_params, TL_model_path)
    
    if RUNTHIS == 2:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['TRD_OG_TL_BASE_ad_rMHA', 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 41
        ident = 'TL_model-TRD_OG_tuned-%s_run-%s' % (city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        multicase(ident, configs_data, run_params, TL_model_path)
        
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['TRD_OG_TL_BASE_ad_rMHA', 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 42
        ident = 'TL_model-TRD_OG_tuned-%s_run-%s' % (city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        multicase(ident, configs_data, run_params, TL_model_path)





    # validation checks.....baseline errors. Tuned model is check against source data 
    if RUNTHIS == 4:
        #SAVE_PATH = 'saved_results/19july/lr-%s_%s_%s_%s-%s/' % (str(lr_not), str(opt), str(batch), city, str(i))
        #PARAM_ST = 'lr-%s_%s_%s_%s-%s' % (str(lr_not), str(opt), str(batch), city, str(i))
        SAVE_PATH = 'saved_results/19july/lr-%s_%s_%s_%s-%s/' % (str(lr_not), str(opt), str(batch), city, str(runindex))
        PARAM_ST                       = 'lr-%s_%s_%s_%s-%s' % (str(lr_not), str(opt), str(batch), city, str(runindex))

        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 5
        test_validation_set_training4(configs_data, SAVE_PATH, city, i, dt_ident, PARAM_ST)

    if RUNTHIS == 5:
        #SAVE_PATH = 'saved_results/19july/lr-%s_%s_%s_%s-%s/' % (str(lr_not), str(opt), str(batch), city, str(i))
        #PARAM_ST = 'lr-%s_%s_%s_%s-%s' % (str(lr_not), str(opt), str(batch), city, str(i))
        SAVE_PATH = 'saved_results/19july/lr-%s_%s_%s_%s-%s/' % (str(lr_not), str(opt), str(batch), city, str(runindex))
        PARAM_ST                       = 'lr-%s_%s_%s_%s-%s' % (str(lr_not), str(opt), str(batch), city, str(runindex))
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 5
        test_validation_set_training41(configs_data, SAVE_PATH, city, i, dt_ident, PARAM_ST)
        test_validation_set_training42(configs_data, SAVE_PATH, city, i, dt_ident, PARAM_ST)
        
    """
    end_script = dt.datetime.now()
    print('Time taken: %s' % str(end_script-start_script))
    print('All Done :) Good Job!')