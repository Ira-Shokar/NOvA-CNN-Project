from methods import *

# model_optimiser options: 'Adam', 'SGD'
train(path = '/home/ishokar/dataframes/',
      train_type = 'descr',
      epochs = 100,
      batch_size = 32,
      dataset_percent = 0.01,
      call_back_patience = 20,
      learning_rate = 0.0001,
      model_optimiser='SGD',
      out_file_name = 'descr_100__32_SGD_12_02')
