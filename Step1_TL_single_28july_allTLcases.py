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





def finetuning_case(ident, configs, runparams, model_path):
    source_model, target_model, lr, lr_not, tl_type, tl_layer = run_params
    
    configs['if_model_image']                   = 0
    configs['loss']                             = 6
    configs['all_layers_neurons']               = 128
    configs['training']['EarlyStop_patience']   = 50
    configs['training']['epochs']               = 50
    configs['optimizer']                        = 'Adam'
    configs[str(configs['optimizer'])]['lr']    = lr
    configs['training']['batch_size']           = 512

    configs['tl_case']['tl_type']               = int(tl_type)

    configs['data']['input_EP_csv_files']       = [target_model + '.csv', 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']
    print('configs[data][input_EP_csv_files] - ', configs['data']['input_EP_csv_files'])
    configs['data']['save_results_dir'] =  'saved_results/27july/%s/lr-%s-%s-%s-SM-%s-TM-%s/' % (str(source_model),
                                                                                                 str(lr_not),
                                                                                                 str(tl_type),
                                                                                                 str("_".join(tl_layer)),
                                                                                                 str(source_model),
                                                                                                 str(target_model)) 
    
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

    stochnumber = 1              # there are 2 stochastic runs for each pretrained 
    source_models = ['M0_Trondheim', 'M0_Oslo', 'M0_Bergen', 'M0_Roros', 'M0_Tromso']
    #source_models = ['M0_Oslo']
    models      = ['M0', 'M1_new', 'M2', 'M3low', 'M3med', 'M3high']
    #models      = ['M1_new']
    cities      = ['Oslo', 'Bergen', 'Trondheim', 'Roros', 'Tromso']
    lrs         = [0.0001] 
    lrs_not     = ['1e-4']                           
    tl_types    = ['6']          
    #tl_layers   = [['dec1', 'Dense-200'],
    #               ['dec1', 'GRU-100']]
    tl_layers   = [['dec1', 'Dense-200']]
    
    allindexs = []
    for source_model in source_models:
        for model in models:
            for city in cities:
                for lr, lr_not in zip(lrs, lrs_not):
                    for tl_type in tl_types:
                        for tl_layer in tl_layers:
                            target_model = '%s_%s' %(model, city)
                            allindexs.append([source_model,
                                            target_model,
                                            lr, lr_not,
                                            tl_type, tl_layer])
    
    run_params = allindexs[runindex]
    source_model, target_model, lr, lr_not, tl_type, tl_layer = run_params

    dt_ident = '_lr-%s-%s-%s-SM-%s-TM-%s' % (str(lr_not), 
                                            str(tl_type), 
                                            str("_".join(tl_layer)),
                                            str(source_model),
                                            str(target_model)
                                            ) + '_' + str(dt.datetime.now().strftime('%d.%H.%f')[:-2])

    """
    # Self test fully trained target model of half of target data
    model_config = 'config_model_RNN_MHA_CIT3.yaml'
    TL_model_path = get_keywords_files('saved_results/27july/pre_trained/%s/' % target_model, 
                                        ['%s_FULL_rMHA' % target_model, 'runNUMBER-%s' % stochnumber], 
                                        'h5')
    print('Self test fully trained target model of half of target data')
    print(TL_model_path)
    allconfigs = [data_config, training_config, model_config]
    configs_data = merge_configs(allconfigs, rl_path='src/configs/')
    configs_data['training']['training_type'] = 5
    ident = 'TL_target_FULL_SELF-TEST_run-%s' % (stochnumber) + dt_ident
    print()
    print('--------------------------**************here begin the TL run for----------------')
    print(TL_model_path)
    print(ident)
    print()
    finetuning_case(ident, configs_data, run_params, TL_model_path)
    
    # Self test half trained target model of half of target data
    model_config = 'config_model_RNN_MHA_CIT3.yaml'
    TL_model_path = get_keywords_files('saved_results/27july/pre_trained/%s/' % target_model, 
                                        ['%s_HALF_rMHA' % target_model, 'runNUMBER-%s' % stochnumber], 
                                        'h5')
    print('Self test half trained target model of half of target data')
    print(TL_model_path)
    allconfigs = [data_config, training_config, model_config]
    configs_data = merge_configs(allconfigs, rl_path='src/configs/')
    configs_data['training']['training_type'] = 5
    ident = 'TL_target_HALF_SELF-TEST_run-%s' % (stochnumber) + dt_ident
    print()
    print('--------------------------**************here begin the TL run for----------------')
    print(TL_model_path)
    print(ident)
    print()
    finetuning_case(ident, configs_data, run_params, TL_model_path)
    """

    # Source model tested on half of target data
    model_config = 'config_model_RNN_MHA_CIT3.yaml'
    TL_model_path = get_keywords_files('saved_results/27july/pre_trained/%s/' % source_model, 
                                        ['%s_FULL_rMHA' % source_model, 'runNUMBER-%s' % stochnumber], 
                                        'h5')
    
    print('Source model tested on half of target data')
    print(TL_model_path)
    allconfigs = [data_config, training_config, model_config]
    configs_data = merge_configs(allconfigs, rl_path='src/configs/')
    configs_data['training']['training_type'] = 5
    ident = 'TL_source-halfTEST_run-%s' % (stochnumber) + dt_ident
    print()
    print('--------------------------**************here begin the TL run for----------------')
    print(TL_model_path)
    print(ident)
    print()
    finetuning_case(ident, configs_data, run_params, TL_model_path)
    
    """
    # Source model tuned using weekly data of target model
    model_config = 'config_model_RNN_MHA_CIT3.yaml'
    TL_model_path = get_keywords_files('saved_results/27july/pre_trained/%s/' % source_model, 
                                        ['%s_FULL_rMHA' % source_model, 'runNUMBER-%s' % stochnumber], 
                                        'h5')
    print('Source model tuned using weekly data of target model')
    print(TL_model_path)
    allconfigs = [data_config, training_config, model_config]
    configs_data = merge_configs(allconfigs, rl_path='src/configs/')
    configs_data['training']['training_type'] = 4
    configs_data['tl_case']['start'] = 0
    configs_data['tl_case']['end']   = 27
    ident = 'TL_source_TUNED_run-%s' % (stochnumber) + dt_ident
    print()
    print('--------------------------**************here begin the TL run for----------------')
    print(TL_model_path)
    print(ident)
    print()
    finetuning_case(ident, configs_data, run_params, TL_model_path)
    """

    end_script = dt.datetime.now()
    print('Time taken: %s' % str(end_script-start_script))
    print('All Done :) Good Job!')