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


def onecase(ident, configs, ifdata, filename1, model_path=None):
    configs['if_model_image'] = 0
    configs['CustomEarlyStopping']['min_error'] = 1000
    configs['CustomEarlyStopping']['check_epoch_index'] = 5
    configs['training']['batch_size'] = 512
    configs['training']['training_type'] = 2
    configs['training']['epochs'] = 65
    configs['training']['num_of_same_run'] = 2
    configs['data']['save_results_dir'] = 'saved_results/27july/pre_trained/%s/' % filename1
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


if __name__ == '__main__':
    data_config = 'config_data.yaml'
    training_config = 'config_training.yaml'

    RUNTHIS = int(sys.argv[1])
    print('Running case: ', RUNTHIS)

    #models      = ['M0', 'M1', 'M2', 'M3low', 'M3med', 'M3high']
    models      = ['M1_new']
    cities    = ['Oslo', 'Bergen', 'Trondheim', 'Roros', 'Tromso']

    filenames = []
    for model in models:
        for city in cities:
            filenames.append('%s_%s' %(model, city))

    filename = filenames[RUNTHIS]

    model_config = 'config_model_RNN_MHA_CIT3.yaml'
    TL_model_path = '----change-----'
    allconfigs = [data_config, training_config, model_config]
    configs_data = merge_configs(allconfigs, rl_path='src/configs/')
    configs_data['data']['half_training'] = 0
    configs_data['loss'] = 1
    configs_data['data']['input_EP_csv_files'] = [filename + '.csv', 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']
    ident = '%s_FULL_rMHA_' % filename
    onecase(ident, configs_data, False, filename, TL_model_path)

    model_config = 'config_model_RNN_MHA_CIT3.yaml'
    TL_model_path = '----change-----'
    allconfigs = [data_config, training_config, model_config]
    configs_data = merge_configs(allconfigs, rl_path='src/configs/')
    configs_data['data']['half_training'] = 1
    configs_data['loss'] = 1
    configs_data['data']['input_EP_csv_files'] = [filename + '.csv', 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']
    ident = '%s_HALF_rMHA_' % filename
    onecase(ident, configs_data, False, filename, TL_model_path)

    print('Finished all cases')
   
    ''' '''