
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


def finetuning_case(ident, configs, runparams, model_path):
    batch, epoch, lr, lr_not, opt, tl_type, tl_layer, city, stoch, runindex = runparams
    
    configs['if_model_image']                   = 0
    configs['loss']                             = 6
    configs['all_layers_neurons']               = 128
    configs['training']['EarlyStop_patience']   = 50
    configs['training']['epochs']               = epoch
    configs['optimizer']                        = opt
    configs[str(configs['optimizer'])]['lr']    = lr
    configs['training']['batch_size']           = batch

    configs['tl_case']['tl_type']               = int(tl_type)

    configs['data']['input_EP_csv_files']       = ['%s_GEO_CH_TL_nohvac_nowindow_ts4.csv' % city, 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']

    configs['data']['save_results_dir'] =  'saved_results/21july/lr-%s_%s_%s_%s-%s-%s_%s/' % (str(lr_not), str(opt), str(batch), city, str(runindex), str(tl_type), str("_".join(tl_layer)))
    
    if configs['tl_case']['tl_type'] == 2 or configs['tl_case']['tl_type'] == 3:
        assert len(tl_layer) == 1, 'tl_layer must have only one value or string if tl_type is 2 or 3'
        tl_layer = tl_layer[0]
        if tl_layer == 'enc':
            configs['tl_case']['some_layer'] = [2, 6, 8, 10, 12, 14]
        elif tl_layer == 'enc1':
            configs['tl_case']['some_layer'] = [2, 6, 8]
        elif tl_layer == 'enc2':
            configs['tl_case']['some_layer'] = [10, 12, 14]
        elif tl_layer == 'dec':
            configs['tl_case']['some_layer'] = [3, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22]
        elif tl_layer == 'dec1':
            configs['tl_case']['some_layer'] = [3, 7, 9]
        elif tl_layer == 'dec2':
            configs['tl_case']['some_layer'] = [11, 13]
        elif tl_layer == 'dec3':
            configs['tl_case']['some_layer'] = [15, 16, 17, 18]
        elif tl_layer == 'dec4':
            configs['tl_case']['some_layer'] = [19, 20, 21, 22]
        else:
            configs['tl_case']['some_layer'] = [int(tl_layer)]

    elif configs['tl_case']['tl_type'] == 4:
        assert len(tl_layer) == 1, 'tl_layer must have only one value or string if tl_type is 4'
        tl_layer = tl_layer[0]
        configs['tl_case']['extra_layer_type'] = tl_layer

    elif configs['tl_case']['tl_type'] == 5 or configs['tl_case']['tl_type'] == 6:
        assert len(tl_layer) == 2, 'tl_layer must have two values or strings if tl_type is 5'
        tl_5_1_layer = tl_layer[0]
        tl_5_2_layer = tl_layer[1]

        configs['tl_case']['tl_5_2_extra_layer_type'] = tl_5_2_layer

        if tl_5_1_layer == 'enc':
            configs['tl_case']['tl_5_1_layer'] = [2, 6, 8, 10, 12, 14]
        elif tl_5_1_layer == 'enc1':
            configs['tl_case']['tl_5_1_layer'] = [2, 6, 8]
        elif tl_5_1_layer == 'enc2':
            configs['tl_case']['tl_5_1_layer'] = [10, 12, 14]
        elif tl_5_1_layer == 'dec':
            configs['tl_case']['tl_5_1_layer'] = [3, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22]
        elif tl_5_1_layer == 'dec1':
            configs['tl_case']['tl_5_1_layer'] = [3, 7, 9]
        elif tl_5_1_layer == 'dec2':
            configs['tl_case']['tl_5_1_layer'] = [11, 13]
        elif tl_5_1_layer == 'dec3':
            configs['tl_case']['tl_5_1_layer'] = [15, 16, 17, 18]
        elif tl_5_1_layer == 'dec4':
            configs['tl_case']['tl_5_1_layer'] = [19, 20, 21, 22]
        else:
            configs['tl_case']['tl_5_1_layer'] = [int(tl_5_1_layer)]

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

    batch_sizes = [512] # [256, 512]
    epochs      = [50] # [25, 50]
    lrs         = [0.0001] # 4
    lrs_not     = ['1e-4']  # 5                         # for tl_type 3 and 4, 1e-5 activates LRS
    optimizers  = ['Adam'] # 2
    #tl_types    = ['2', '3']          # 3 / 1e-5 is actually with LRS. 1e-4 is without LRS
    tl_types    = ['5','6']          # 3 or 4 / 1e-5 is actually with LRS. 1e-4 is without LRS
                                # for 5 / 1e-5 is actually with LRS. 1e-4 is without LRS
                                # for 6 / 1e-4 with LRS
    #tl_layers   = ['2', '3', '12', '13', '16']  #['22', '20', '19', '18', '15', '14', '11', '10', '7', '6']
    #tl_layers   = ['enc1', 'enc2', 'dec1', 'dec2', 'dec3', 'dec4']
    #tl_layers   = ['Dense-relu', 'Dense-sigmoid', 'Dense-tanh', 'LSTM', 'GRU', 'MHA']

    tl_layers   = ['Dense-25', 'Dense-50', 'Dense-100', 'Dense-200', 'Dense-400', 
                   'GRU-25',   'GRU-50',   'GRU-100',   'GRU-200',   'GRU-400',
                   'MHA-4_1',  'MHA-4_2',  'MHA-8_1',   'MHA-8_2',   'MHA_16_1', 'MHA_16_2']
    tl_layers   = [[i] for i in tl_layers]

    tl_layers  = [['dec', 'Dense-200'],
                  ['dec', 'GRU-100'],
                  ['dec', 'MHA-8_1'],
                  ['dec1', 'Dense-200'],
                  ['dec1', 'GRU-100'],
                  ['dec1', 'MHA-8_1'],
                  ['dec2', 'Dense-200'],
                  ['dec2', 'GRU-100'],
                  ['dec2', 'MHA-8_1'],
                  ['dec3', 'Dense-200'],
                  ['dec3', 'GRU-100'],
                  ['dec3', 'MHA-8_1'],
                  ['dec4', 'Dense-200'],
                  ['dec4', 'GRU-100'],
                  ['dec4', 'MHA-8_1']]
    
    tl_layers  = [['dec1', 'Dense-200'],
                  ['dec1', 'GRU-100']]
    
    allindexs = []
    total_runs = 5

    for batch, epoch in zip(batch_sizes, epochs):
        for lr, lr_not in zip(lrs, lrs_not):
            for opt in optimizers:
                for tl_type in tl_types:
                    for tl_layer in tl_layers:
                        for run in range(1, total_runs+1):
                            allindexs.append([batch, epoch, lr, lr_not, opt, tl_type, tl_layer, city, i, run])
    
    #run_params = [512, 25, 0.0001, '1e-5', 'Adam', '3', ['dec3']]
    run_params = allindexs[runindex-1]
    batch, epoch, lr, lr_not, opt, tl_type, tl_layer, city, i, runindex = run_params

    dt_ident = '_lr-%s_%s_%s_%s-%s-%s_%s' % (str(lr_not), str(opt), str(batch), city, str(runindex), str(tl_type), str("_".join(tl_layer))) + '_' + str(dt.datetime.now().strftime('%d.%H.%f')[:-2])

   
    if RUNTHIS == 1:
        """"""
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
        finetuning_case(ident, configs_data, run_params, TL_model_path)
        
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
        finetuning_case(ident, configs_data, run_params, TL_model_path)
        
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
        finetuning_case(ident, configs_data, run_params, TL_model_path)
    
    # tune model for monthly and weekly
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
        finetuning_case(ident, configs_data, run_params, TL_model_path)
        
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
        finetuning_case(ident, configs_data, run_params, TL_model_path)





    # VALIDATION CHECKS....BASELINE ERROR. Tuned model is check against source data 
    # running week
    if RUNTHIS == 11:
        #SAVE_PATH = 'saved_results/21july/lr-%s_%s_%s_%s-%s-%s_%s/' % (str(lr_not), str(opt), str(batch), city, str(i), str(tl_type), str("_".join(tl_layer)))
        #PARAM_ST                       = 'lr-%s_%s_%s_%s-%s-%s_%s' % (str(lr_not), str(opt), str(batch), city, str(i), str(tl_type), str("_".join(tl_layer)))
        SAVE_PATH  = 'saved_results/21july/lr-%s_%s_%s_%s-%s-%s_%s/' % (str(lr_not), str(opt), str(batch), city, str(runindex), str(tl_type), str("_".join(tl_layer)))
        PARAM_ST                        = 'lr-%s_%s_%s_%s-%s-%s_%s' % (str(lr_not), str(opt), str(batch), city, str(runindex), str(tl_type), str("_".join(tl_layer)))

        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 5
        test_validation_set_training4(configs_data, SAVE_PATH, city, i, dt_ident, PARAM_ST)

    #monthly, weekly
    if RUNTHIS == 2:
        #SAVE_PATH  = 'saved_results/21july/lr-%s_%s_%s_%s-%s-%s_%s/' % (str(lr_not), str(opt), str(batch), city, str(i), str(tl_type), str("_".join(tl_layer)))
        #PARAM_ST                       =  'lr-%s_%s_%s_%s-%s-%s_%s' % (str(lr_not), str(opt), str(batch), city, str(i), str(tl_type), str("_".join(tl_layer)))
        SAVE_PATH   = 'saved_results/21july/lr-%s_%s_%s_%s-%s-%s_%s/' % (str(lr_not), str(opt), str(batch), city, str(runindex), str(tl_type), str("_".join(tl_layer)))
        PARAM_ST                        =  'lr-%s_%s_%s_%s-%s-%s_%s' % (str(lr_not), str(opt), str(batch), city, str(runindex), str(tl_type), str("_".join(tl_layer)))
        
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 5
        test_validation_set_training41(configs_data, SAVE_PATH, city, i, dt_ident, PARAM_ST)
        test_validation_set_training42(configs_data, SAVE_PATH, city, i, dt_ident, PARAM_ST)
        
    
    end_script = dt.datetime.now()
    print('Time taken: %s' % str(end_script-start_script))
    print('All Done :) Good Job!')