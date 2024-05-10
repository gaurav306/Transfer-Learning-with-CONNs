__author__ = "Gaurav Chaudhary"
__copyright__ = "Gaurav Chaudhary 2022"
__version__ = "1.0.0"
__license__ = "MIT"
import os

import datetime as dt
import sys
from src.configs.configs_init import merge_configs
from src.data_processor import DataClass
from src.main_run import Main_Run_Class

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def onecase(ident, configs, ifdata, model_path=None):
    configs['if_model_image'] = 0
    configs['CustomEarlyStopping']['min_error'] = 1000
    configs['CustomEarlyStopping']['check_epoch_index'] = 5
    configs['training']['batch_size'] = 512
    configs['training']['training_type'] = 2
    configs['training']['epochs'] = 5
    configs['training']['num_of_same_run'] = 1
    configs['training']['cosine_start_lr'] = 0.05
    configs['data']['save_results_dir'] = 'saved_results/19july/pre_trained/'
    configs['optimizer'] = 'Adam'

    configs['all_layers_neurons'] = 128
    ident = ident + str(dt.datetime.now().strftime('%d.%M.%S.%f')[:-3])
    print('Current run case: ', ident)
    if ifdata:    
        print('Opening data class')
        data_class = DataClass(ident, configs)
    print('Opening training pipeline class')
    run_pipeline = Main_Run_Class(ident, configs, model_path)
    print('%s finished' % ident)

def multicase(ident, configs, ifdata, run_index, model_path=None):
    configs['if_model_image'] = 0
    configs['CustomEarlyStopping']['min_error'] = 1000
    configs['CustomEarlyStopping']['check_epoch_index'] = 10
    configs['training']['training_type'] = 2
    
    dt_ident = ''
    if ifdata:
        print('Opening data class')
        data_class = DataClass(ident, configs)
    print('Opening training pipeline class')

    neurons     = [256, 512]
    drs         = [0.2]
    rnns        = ['GRU']
    batch_sizes = [256, 512]
    cosine_lrs  = [0.05, 0.001]
    allindexs = []
    for nn in neurons:
        for dr in drs:
            for rnn in rnns:
                for batch_size in batch_sizes:
                    for cosine_lr in cosine_lrs:
                        allindexs.append([nn, dr, rnn, batch_size, cosine_lr])
    nn  = allindexs[run_index][0]
    dr  = allindexs[run_index][1]
    rnn = allindexs[run_index][2]
    bs  = allindexs[run_index][3]
    clr = allindexs[run_index][4]

    configs['all_layers_neurons'] = nn
    configs['all_layers_dropout'] = dr
    configs['encoder']['RNN_block_input']['rnn_type'] = rnn
    configs['decoder']['RNN_block_input']['rnn_type'] = rnn
    configs['decoder']['RNN_block_output']['rnn_type'] = rnn
    configs['training']['cosine_start_lr'] = clr
    configs['training']['batch_size'] = bs
    ident = 'UT_index-%s_nn-%s_dr-%s_rnn-%s_clr-%s_bs-%s_' % (str(run_index), str(nn), str(dr), str(rnn), str(clr), str(bs))

    
    ident = ident + dt_ident
    print('Current run case: ', ident)
    run_pipeline = Main_Run_Class(ident, configs, model_path)
    print('%s finished' % ident)



if __name__ == '__main__':
    """
    Trains base model for original TRD data. Trains GEO_CH model for BERGEN and OSLO data for full and half training.
    """
    data_config = 'config_data.yaml'
    training_config = 'config_training.yaml'

    RUNTHIS = int(sys.argv[1])
    #RUNTHIS = 1
    #runindex = int(sys.argv[1])
    runindex = 0

    if RUNTHIS == 1:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = '----change-----'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['data']['half_training'] = 0
        configs_data['loss'] = 1
        configs_data['data']['input_EP_csv_files'] = ['BERGEN_GEO_CH_TL_nohvac_nowindow_ts4.csv', 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']
        ident = 'BERGEN_GEO_CH_TL_FULL_rMHA_'
        onecase(ident, configs_data, False, TL_model_path)

    if RUNTHIS == 2:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = '----change-----'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['data']['half_training'] = 1
        configs_data['loss'] = 1
        configs_data['data']['input_EP_csv_files'] = ['BERGEN_GEO_CH_TL_nohvac_nowindow_ts4.csv', 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']
        ident = 'BERGEN_GEO_CH_TL_HALF_rMHA_'
        onecase(ident, configs_data, False, TL_model_path)
    """
    if RUNTHIS == 3:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = '----change-----'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['data']['half_training'] = 0
        configs_data['loss'] = 1
        configs_data['data']['input_EP_csv_files'] = ['TRD_OG_TL_nohvac_nowindow_ts4.csv', 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']
        ident = 'TRD_OG_TL_BASE_rMHA_'
        onecase(ident, configs_data, False, TL_model_path)

    if RUNTHIS == 23:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = '----change-----'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['data']['half_training'] = 0
        configs_data['loss'] = 1
        ident = ''
        multicase(ident, configs_data, False, runindex, TL_model_path)

    if RUNTHIS == 0:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = '----change-----'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['data']['half_training'] = 0
        configs_data['loss'] = 1
        ident = 'TRD_noCTRL_ts4_'
        onecase(ident, configs_data, False, TL_model_path)

    """
    """
    if RUNTHIS == 3:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = '----change-----'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['data']['half_training'] = 0
        configs_data['loss'] = 1
        configs_data['data']['input_EP_csv_files'] = ['OSLO_GEO_CH_TL_nohvac_nowindow_ts4.csv', 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']
        ident = 'OSLO_GEO_CH_TL_FULL_rMHA_'
        onecase(ident, configs_data, False, TL_model_path)

    if RUNTHIS == 4:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = '----change-----'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['data']['half_training'] = 1
        configs_data['loss'] = 1
        configs_data['data']['input_EP_csv_files'] = ['OSLO_GEO_CH_TL_nohvac_nowindow_ts4.csv', 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']
        ident = 'OSLO_GEO_CH_TL_HALF_rMHA_'
        onecase(ident, configs_data, False, TL_model_path)
    """
    


    print('Finished all cases')
   
    ''' '''