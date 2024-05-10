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
        raise ValueError('No file found for keywords: {}'.format(keywords))
        return None
    else:
        return matching_files[0]


def make_boxplot3_multi_npy(folder_path, city, stoch, ST):
    coloumn_index_names = ['CVRMSE(%)', 'RMSE', 'MAE', 'MSE', 'Rsquared', 'DTW.Distance']
    metrics_indexes_to_plots = [2,5]
    
    fig = plt.figure(figsize=(5*len(metrics_indexes_to_plots),20))
    fig.subplots_adjust(wspace=0.1, hspace=0.5)    

    def makedata(city, i, ST, coloumn_index):
        data_all = pd.DataFrame()
        all_col_names = []

        filename = get_keywords_files(folder_path, ['TL_model-%s_GEO_CH_HALF' % city, '_cvrmse_r2_nmbe_bytimestep.npy', 'run-%s%s'% (i,ST)], 'npy')
        if filename is None:
            temp_data = pd.Series(np.zeros((96,)))
        else:
            temp_data = pd.Series(np.load(filename)[:,0,coloumn_index])
        #print(temp_data.shape)
        all_col_names.append('Fresh model trained\nHALF %s data' % city)
        data_all = pd.concat([data_all, temp_data], axis=1)

        for ix in range(0,27):
            filename = get_keywords_files(folder_path, ['TL_model-TRD_OG_tuned-%s' % city, 'week-%s_cvrmse_r2_nmbe_bytimestep' % str(ix), 'run-%s%s'% (i,ST)], 'npy')
            if filename is None:
                temp_data = pd.Series(np.zeros((96,)))
            else:
                temp_data = pd.Series(np.load(filename)[:,0,coloumn_index])
            all_col_names.append('0 to -%s week/s of %s data' % (str(ix), city))
            #all_col_names.append('TRD_model tuned with \n0 to -%s week/s of %s data' % (str(ix), city))
            data_all = pd.concat([data_all, temp_data], axis=1)    
        '''
        for ix in range(0,26):
            filename = get_keywords_files(folder_path, ['TL_model-TRD_OG_tuned-%s' % city, 'weekly-%s_cvrmse_r2_nmbe_bytimestep' % str(ix), 'run-%s%s'% (i,ST)], 'npy')
            if filename is None:
                temp_data = pd.Series(np.zeros((96,)))
            else:
                temp_data = pd.Series(np.load(filename)[:,0,coloumn_index])
            all_col_names.append('only -%s(th) week/s of %s data' % (str(ix+1), city))
            data_all = pd.concat([data_all, temp_data], axis=1)    

        for ix in range(0,6):
            filename = get_keywords_files(folder_path, ['TL_model-TRD_OG_tuned-%s' % city, 'month-%s_cvrmse_r2_nmbe_bytimestep' % str(ix), 'run-%s%s'% (i,ST)], 'npy')
            if filename is None:
                temp_data = pd.Series(np.zeros((96,)))
            else:
                temp_data = pd.Series(np.load(filename)[:,0,coloumn_index])
            all_col_names.append('only -%s(th) month/s of %s data' % (str(ix+1), city))
            data_all = pd.concat([data_all, temp_data], axis=1)    
        '''
        filename = get_keywords_files(folder_path, ['TL_model-%s_GEO_CH_FULL' % city, '_cvrmse_r2_nmbe_bytimestep', 'run-%s%s'% (i,ST)], 'npy')
        if filename is None:
            temp_data = pd.Series(np.zeros((96,)))
        else:
            temp_data = pd.Series(np.load(filename)[:,0,coloumn_index])
        all_col_names.append('Fresh model trained\nFULL %s data' % city)
        data_all = pd.concat([data_all, temp_data], axis=1)

        data_all.columns = all_col_names
        return data_all
    
    for i_plot, coloumn_index in enumerate(metrics_indexes_to_plots):
        #coloumn_index = 0
        df = makedata(city, stoch, ST, coloumn_index)
        ax = fig.add_subplot(1,len(metrics_indexes_to_plots),i_plot+1)
        vals, names, xs, means = [],[],[], []
        for ix, col in enumerate(df.columns):
            vals.append(df[col].values)
            names.append(col)
            xs.append(np.random.normal(ix + 1, 0.06, df[col].values.shape[0]))
            means.append(np.mean(df[col].values))
        ax.boxplot(vals, labels=names, notch=True, showfliers=False, vert=False, whis=[5, 95], widths=0.3, meanline=True, showmeans=True) #, medianprops={'linewidth': 2, 'color': 'red'}, meanprops={'linewidth': 2, 'color': 'blue'}
        for x, val in zip(xs, vals):
            ax.scatter( val, x, alpha=0.4, marker='.', s=20)

        ax.plot(means[1:28], range(2, 29), lw=1, color='red', linestyle='--')
        #ax.plot(means[28:54], range(29, 55), lw=1, color='green', linestyle='--')
        #ax.plot(means[54:-1], range(55, len(means)), lw=1, color='black', linestyle='--')

        ax.grid(True, alpha=0.4)
        ax.set_title('TRD>%s - Transfer learning on\ndifferent building-Model:RNN-MHA\nrun-%s_%s\n%s' % (city, stoch, ST, coloumn_index_names[coloumn_index]), fontsize=10)
        ax.set_xlabel(coloumn_index_names[coloumn_index])
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in', pad=5)
        ax.tick_params(axis='y', which='major', labelsize=8)
        ax.tick_params(axis='x', which='major', labelsize=10)
        #ax.set_xlim(0, 0.15)
        if i_plot > 0:
            ax.set_yticklabels([])

    plt.savefig(folder_path + '%s_GEO_CH_rMHA_run-%s_%s.png' % (city, stoch, ST), bbox_inches='tight', dpi=600)
    print('saved %s_GEO_CH_rMHA_run-%s_%s.png' % (city, stoch, ST))
    plt.close()


def onecase_TL(ident, ident2, configs, ifdata, opti, model_path = None):
    
    configs['if_model_image']           = 0
    configs['loss']                     = 6
    configs['training']['batch_size']   = 512
    configs['all_layers_neurons']       = 128
    configs['training']['epochs']       = 25
    configs['training']['EarlyStop_patience'] = 15
    
    configs['data']['save_results_dir'] = ident2
    
    if opti == 1:
        configs['optimizer']                        = 'SGD'
        configs[str(configs['optimizer'])]['lr']    = 0.005
    if opti == 2:
        configs['optimizer']                        = 'Adam'
        configs[str(configs['optimizer'])]['lr']    = 0.00005
    
    if ifdata:    
        print('Opening data class')
        data_class = DataClass(ident, configs)
    print('Opening training pipeline class')
    run_pipeline = Main_Run_Class(ident, configs, model_path)

if __name__ == '__main__':

    start_script = dt.datetime.now()
    data_config         = 'config_data.yaml'
    training_config     = 'config_training.yaml'

    #RUNTHIS = 1
    RUNTHIS = int(sys.argv[1])
    opti    = int(sys.argv[2])
    
    TL_CITIES = ['BERGEN', 'OSLO']
    TL_STOCHASTICS = ['1', '2']
    city, i = 'BERGEN', '1'

    if opti == 1:
        TL_SETTINGS = ['lr-5e3_SGD']
    elif opti == 2:
        TL_SETTINGS = ['lr-5e5_Adam_512']

    SAVE_PATH = 'saved_results/19july/%s_BERGEN/' % TL_SETTINGS[0]
    

    print('***********************************************************************************************')
    print(city, i)
    print('***********************************************************************************************')
    DATA_INPUT = ['%s_GEO_CH_TL_nohvac_nowindow_ts4.csv' % city, 'TRD_GEO_CH_TL_nohvac_nowindow_ts4.csv']
    dt_ident = TL_SETTINGS[0] + '_' + str(dt.datetime.now().strftime('%d.%H.%f')[:-2])
    
    if RUNTHIS == 1:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['%s_GEO_CH_TL_FULL_ad_rMHA' % city, 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 5
        configs_data['data']['input_EP_csv_files'] = DATA_INPUT
        ident = 'TL_model-%s_GEO_CH_FULL_test-%s_run-%s' % (city, city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        onecase_TL(ident, SAVE_PATH, configs_data, False, opti, TL_model_path)
        
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['%s_GEO_CH_TL_HALF_ad_rMHA' % city, 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 5
        configs_data['data']['input_EP_csv_files'] = DATA_INPUT
        ident = 'TL_model-%s_GEO_CH_HALF_test-%s_run-%s' % (city, city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        onecase_TL(ident, SAVE_PATH, configs_data, False, opti, TL_model_path)

        #if RUNTHIS == 2:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['TRD_OG_TL_BASE_ad_rMHA', 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 4
        configs_data['data']['input_EP_csv_files'] = DATA_INPUT
        configs_data['tl_case']['start'] = 0
        configs_data['tl_case']['end']   = 7
        ident = 'TL_model-TRD_OG_tuned-%s_run-%s' % (city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        onecase_TL(ident, SAVE_PATH, configs_data, False, opti, TL_model_path)
    
    if RUNTHIS == 2:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['TRD_OG_TL_BASE_ad_rMHA', 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 4
        configs_data['data']['input_EP_csv_files'] = DATA_INPUT
        configs_data['tl_case']['start'] = 7
        configs_data['tl_case']['end']   = 13
        ident = 'TL_model-TRD_OG_tuned-%s_run-%s' % (city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        onecase_TL(ident, SAVE_PATH, configs_data, False, opti, TL_model_path)
    
    if RUNTHIS == 3:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['TRD_OG_TL_BASE_ad_rMHA', 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 4
        configs_data['data']['input_EP_csv_files'] = DATA_INPUT
        configs_data['tl_case']['start'] = 13
        configs_data['tl_case']['end']   = 18
        ident = 'TL_model-TRD_OG_tuned-%s_run-%s' % (city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        onecase_TL(ident, SAVE_PATH, configs_data, False, opti, TL_model_path)
    
    if RUNTHIS == 4:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['TRD_OG_TL_BASE_ad_rMHA', 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 4
        configs_data['data']['input_EP_csv_files'] = DATA_INPUT
        configs_data['tl_case']['start'] = 18
        configs_data['tl_case']['end']   = 22
        ident = 'TL_model-TRD_OG_tuned-%s_run-%s' % (city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        onecase_TL(ident, SAVE_PATH, configs_data, False, opti, TL_model_path)
    
    if RUNTHIS == 5:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['TRD_OG_TL_BASE_ad_rMHA', 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 4
        configs_data['data']['input_EP_csv_files'] = DATA_INPUT
        configs_data['tl_case']['start'] = 22
        configs_data['tl_case']['end']   = 27
        ident = 'TL_model-TRD_OG_tuned-%s_run-%s' % (city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        onecase_TL(ident, SAVE_PATH, configs_data, False, opti, TL_model_path)
    
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        make_boxplot3_multi_npy(SAVE_PATH, city, i, TL_SETTINGS[0])

        '''
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['TRD_OG_TL_BASE', 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 41
        configs_data['data']['input_EP_csv_files'] = DATA_INPUT
        ident = 'TL_model-TRD_OG_tuned-%s_run-%s' % (city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        onecase_TL(ident, SAVE_PATH, configs_data, False, opti, TL_model_path)
        
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = get_keywords_files('saved_results/19july/pre_trained/', ['TRD_OG_TL_BASE', 'runNUMBER-%s' % i], 'h5')
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        configs_data['training']['training_type'] = 42
        configs_data['data']['input_EP_csv_files'] = DATA_INPUT
        ident = 'TL_model-TRD_OG_tuned-%s_run-%s' % (city, i) + dt_ident
        print()
        print('--------------------------**************here begin the TL run for----------------')
        print(TL_model_path)
        print(ident)
        print()
        onecase_TL(ident, SAVE_PATH, configs_data, False, opti, TL_model_path)
        '''
        ''''''
    if RUNTHIS == 0:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')
        make_boxplot3_multi_npy(SAVE_PATH, city, i, TL_SETTINGS[0])
    
    end_script = dt.datetime.now()
    print('Time taken: %s' % str(end_script-start_script))
    print('All Done :) Good Job!')