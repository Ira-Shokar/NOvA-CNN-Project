from scripts.methods import *

#function paramemters

path = '/home/ishokar/dataframes/'

#data options: 'both', 'genie', 'gibuu'
data = 'both'
#model_type options: 'default', 'descr'
model_type = 'descr'
#output options: 'default'
output = 'default'

file= '/weights_train_100_descr_32_sgd_22_02.h5'
file_name = 'default_TOboth_equal_1'

dataset_percent = 0.001

test(file, path, file_name, dataset_percent, data, model_type, output)
