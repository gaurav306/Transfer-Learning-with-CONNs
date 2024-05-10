import numpy as np
from numpy import load
from typing import *

class Data_Reader_Class():
	def __init__(self,ident,configs):
		self.csv_files 				= configs['data']['input_EP_csv_files']

		self.configs 				= configs
		self.ident 					= ident
		self.data_split 			= self.configs['data']['data_split']
		self.DATAFOLDER 			= self.configs['data']['npy_save_path']
		self.IFDEBUG_CODE 			= self.configs['data_type_code']

	def __call__(self):
		if self.configs['data']['data_split']=='1a':
			data_all = self._readdata_1a()
		if self.configs['data']['data_split'] in ['2a','2b','3a']:
			data_all = self._readdata_2a_2b_3a()
		return data_all


	def _readdata_2a_2b_3a(self) -> List[np.ndarray]:
		"""Reads the data from the npy files and returns the data in the form of a list of numpy arrays
		
		Returns:
			list -- [train_X_past, train_X_future, train_Y_future, validate_X_past, validate_X_future, validate_Y_future, test_X_past, test_X_future, test_Y_future]

		train_X_future, validate_X_future, test_X_future are lists of numpy arrays. 
		
		Each numpy array is of shape (number of samples, number of future cells, number of features)
		"""	

		train_col = self.configs['data']['data_split_%s' % self.configs['data']['data_split']][0]
		valid_col = self.configs['data']['data_split_%s' % self.configs['data']['data_split']][1]
		test_col  = self.configs['data']['data_split_%s' % self.configs['data']['data_split']][2]


		filename = self.DATAFOLDER + self.csv_files[train_col]
		train_X_past 		= load(filename+'%s_X_past_train.npy' %(self.IFDEBUG_CODE))
		train_X_future 		= load(filename+'%s_X_future_train.npy' %(self.IFDEBUG_CODE))
		train_Y_future	 	= load(filename+'%s_Y_future_train.npy' %(self.IFDEBUG_CODE))



		filename = self.DATAFOLDER + self.csv_files[valid_col]
		validate_X_past 	= load(filename+'%s_X_past_validate.npy' %(self.IFDEBUG_CODE))  
		validate_X_future 	= load(filename+'%s_X_future_validate.npy' %(self.IFDEBUG_CODE))
		validate_Y_future 	= load(filename+'%s_Y_future_validate.npy' %(self.IFDEBUG_CODE))  



		filename = self.DATAFOLDER + self.csv_files[test_col]
		test_X_past 		= load(filename+'%s_X_past_test.npy' %(self.IFDEBUG_CODE))  
		test_X_future 		= load(filename+'%s_X_future_test.npy' %(self.IFDEBUG_CODE))
		test_Y_future 		= load(filename+'%s_Y_future_test.npy' %(self.IFDEBUG_CODE))  

		alldata = [train_X_past, 	train_X_future,		train_Y_future,
				validate_X_past, validate_X_future, 	validate_Y_future,
				test_X_past, 	test_X_future, 		test_Y_future]

		return alldata

	def _readdata_1a(self) -> List[np.ndarray]:
		"""Reads the data from the npy files and returns the data in the form of a list of numpy arrays
		
		Returns:
			list -- [train_X_past, train_X_future, train_Y_future, validate_X_past, validate_X_future, validate_Y_future, test_X_past, test_X_future, test_Y_future]

		train_X_future, validate_X_future, test_X_future are lists of numpy arrays. 
		
		Each numpy array is of shape (number of samples, number of future cells, number of features)
		"""	

		train_col = self.configs['data']['data_split_%s' % self.configs['data']['data_split']][0]

		filename = self.DATAFOLDER + self.csv_files[train_col]
		train_X_past 		= load(filename+'%s_X_past_train.npy' %(self.IFDEBUG_CODE))
		train_X_future 		= load(filename+'%s_X_future_train.npy' %(self.IFDEBUG_CODE))
		train_Y_future	 	= load(filename+'%s_Y_future_train.npy' %(self.IFDEBUG_CODE))


		alldata = [train_X_past, 	train_X_future,		train_Y_future]

		return alldata