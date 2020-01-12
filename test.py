from functions import *
from methods import *

def test_generator(batch_size, dataset):
    batch_images = np.zeros((batch_size, 2, 80, 100))
    size = dataset.shape[0]
    while True:
        for i in range(batch_size):
            index= random.randint(0,size-1)
            while index in index_list:
                index= random.randint(1,size)-1
            index_list.append(index)
            row = dataset.loc[index]
            file= row['file']
            images  = image(maps(file)[row['train_index']])
            images= (images - np.min(images))/ (np.max(images) - np.min(images))
            batch_images[i] = images

        yield batch_images

def test(weights_file, path, batch_no):
    df8 = open_df(path, 8)
    df9 = open_df(path, 9)
    test = [df8, df9]

    df_test = pd.concat(test)
    df_test.index = range(len(df_test['file']))
    steps_per_epoch = round(len(df_test['file'])/(batch_no+1))

    model = mobilenetv2.MobileNetV2(input_shape=((2, 80, 100),), classes=3)
    model.load_weights('/home/ishokar/december_test/' + weights_file)

    probabilities = model.predict_generator(test_generator(batch_no, df_test), steps = steps_per_epoch)
    test_labels_list = test_labels(df_test, index_list)

    with open('probabilities_{}.pkl'.format(i[:-3]),'wb') as f1:
                    pkl.dump(probabilities, f1)

    with open('test_labels_list_{}.pkl'.format(i[:-3]),'wb') as f1:
                    pkl.dump(test_labels_list, f1)


path = '/home/ishokar/december_test/output/'
batch_no = 32
index_list = []
for i in ['weights_32_2_LR_0.0001.h5']:
    test(i, path, batch_no)
