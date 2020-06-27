from scripts.methods import *

# model_optimiser options: 'Adam', 'SGD'

train(train_type = 'dann',
      epochs= 100,
      batch_size = 32,
      dataset_percent = 0.9,
      call_back_patience = 10,
      learning_rate = 0.001,
      DANN_strength = 0.1,
      model_optimiser='Adam',
      out_file_name = '32_Adam_dann_0.5_02_03')
