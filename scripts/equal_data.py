from functions import *
path = '/home/ishokar/dataframes/'
df1 = open_df(path, 1)
df2 = open_df(path, 2)
df3 = open_df(path, 3)
df4 = open_df(path, 4)
df5 = open_df(path, 5)
df6 = open_df(path, 6)
df2_ = open_df_gibuu(path, 2)
train = [df1, df2, df3, df4, df5, df6, df2_]
df_train = pd.concat(train)
dataset_len = len(df_train['file'])
print(dataset_len)
df_train.index = range(dataset_len)


df7 = open_df(path, 7)
df8 = open_df(path, 8)
df14_ = open_df_gibuu(path, 14)

val = [df7, df8, df14_]

df_val = pd.concat(val)
dataset_len = len(df_val['file'])
print(dataset_len)
df_val.index = range(dataset_len)

df9 = open_df(path, 9)
df10 = open_df(path, 10)
df11 = open_df(path, 11)

df3_ = open_df_gibuu(path, 3)

test= [df9, df10, df11, df3_]
df_test = pd.concat(test)
dataset_len = len(df_test['file'])
print(dataset_len)
df_test.index = range(dataset_len)

outpath = '/home/ishokar/dataframes'

df1 = equal_data_generator(df_train,'train')
df2 = equal_data_generator(df_train, 'train')
df3 = equal_data_generator(df_train, 'train')

df = pd.concat([df1, df2, df3])
with open(out_path + '/df_{}.pkl'.format(train_equal),'wb') as f1:
    pkl.dump(df, f1)

df1 = equal_data_generator(df_val,'val')
df2 = equal_data_generator(df_val, 'val')
df3 = equal_data_generator(df_val, 'val')

df = pd.concat([df1, df2, df3])
with open(out_path + '/df_{}.pkl'.format(val_equal),'wb') as f1:
    pkl.dump(df, f1)

df1 = equal_data_generator(df_test,'test')
df2 = equal_data_generator(df_test, 'test')
df3 = equal_data_generator(df_test, 'test')

df = pd.concat([df1, df2, df3])
with open(out_path + '/df_{}.pkl'.format(test_equal),'wb') as f1:
    pkl.dump(df, f1)
