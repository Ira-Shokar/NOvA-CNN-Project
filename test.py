from methods import *

#function paramemters

path = '/home/ishokar/dataframes/'

data = 'both'
model_type = 'descr'
output = 'default'

file= '/weights_train_100_descr_32_sgd_22_02.h5'
file_name = 'default_TOboth_equal_1'

dataset_percent = 0.001

test(file, path, file_name, dataset_percent, data, model_type, output)
