from functions import *
from mobilenetv2 import *
from keras.models import Model
from itertools import islice


def cuts(path):
    """cuts
    Function that takes in an hdf5 file and applied multiple functions to :
    nu mu cuts defined - "CAFAna/Cuts/NueCuts2017.h"
        cuts used - kNue2017NDFiducial && kNue2017NDContain && kNue2017NDFrontPlanes
    """
    #import files
    file_dir = get_files(path)

    #split input files into batches of 50 files

    batches = [file_dir[i:i+100] for i in range(0, len(file_dir), 100)]
    batch_no = 0

    #for set of 50 files
    for i in batches:
        batch_no+=1

        dataframes = []

        #for file in batch
        for j in i:

            try :
                hdf, file, Train_Params = data(path + '/' + j)

                ######### Cuts #############################################################

                cut_arr_mu = mu_cuts(hdf, file)
                dataframes.append( apply_cuts(Train_Params, cut_arr_mu, file) )

                cut_arr_e = e_cuts(hdf, file)
                dataframes.append( apply_cuts(Train_Params, cut_arr_e, file))

            except OSError:
                pass

        #### Save files #########################################################################

        out_path = '/home/ishokar/dataframes/'

        df = pd.concat(dataframes)
        df.index = range(len(df['file']))

        with open(out_path + '/df_{}.pkl'.format(batch_no),'wb') as f1:
            pkl.dump(df, f1)







def train(train_type = 'default',
          epochs= 200,
          batch_size = 32,
          dataset_percent = 0.8,
          call_back_patience = 10,
          learning_rate = 0.001,
          DANN_strength = 0.1, 
          model_optimiser='SGD',
          out_file_name = '32_SGD'):

    """train_mobnetmodelv2
    Function that compiles and trains the mobilenet network
    # Arguments/Hyperparamaters
        epochs: number of epochs,
        batch_size : batch size,
        learning_rate : learning rate,
        call_back_patience : number of epochs patience before stopping training if no improvement takes place,
        save_best : saves weights of the best model based on validation loss
    # Returns
	mobnetmodel: trained model
        history : training statistics
    """
    path = '/home/ishokar/dataframes/'
    df_train = open_df(path, 'train_equal')
    df_train.index = range(len(df_train['file']))
    steps_per_epoch = round(len(df_train['file'])/(batch_size)*dataset_percent)
    
    df_val = open_df(path, 'val_equal')
    df_val.index = range(len(df_val['file']))
    val_steps_per_epoch = round(len(df_val['file'])/(batch_size)*dataset_percent)

    if model_optimiser == 'SGD':
        opt= SGD(lr=learning_rate, momentum=0.9)
    elif model_optimiser == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        print('Not valid model_optimiser name')

    callbacks = [EarlyStopping(monitor='accuracy', patience=call_back_patience)]

    if train_type == 'dann':
        mobnetmodel = MobileNetV2_DANN(input_shape=((2, 80, 100),), classifier_classes=3, descriminator_classes = 2, DANN_strength = DANN_strength)
        mobnetmodel.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

    elif train_type== 'default':
        mobnetmodel = mobilenetv2.MobileNetV2(input_shape=((2, 80, 100),), classes=3)
        mobnetmodel.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

    elif train_type == 'descr':
        mobnetmodel = mobilenetv2.MobileNetV2(input_shape=((2, 80, 100),), classes=2)
        mobnetmodel.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])

    history = mobnetmodel.fit_generator(generator=generator(batch_size, steps_per_epoch, df_train, model = train_type),
                                        steps_per_epoch= steps_per_epoch,
                                        validation_data= generator(batch_size, val_steps_per_epoch, df_val, model = train_type),
                                        validation_steps= val_steps_per_epoch,
                                        epochs=epochs,
                                        callbacks = callbacks)

    mobnetmodel.save_weights("weights_{}.h5".format(out_file_name))
    history.model = None
    pkl.dump(history, open( "history_{}.pkl".format(out_file_name), "wb" ))

    return mobnetmodel, history







def test(weights_file, path, name, dataset_percent = 0.1, data = 'both', model_type = 'default', output = 'default'):

    probabilities =[]
    layer_nodes = []    
    batch_no = 32

    df_test = open_df(path, 'test_equal')
    df_test.index = range(len(df_test['file']))
    df_test =df_test.sample(frac=1).reset_index(drop=True)
    steps_per_epoch = round(len(df_test['file'])*dataset_percent)
    
    columns = df_test.columns    
    df_row = pd.DataFrame(columns = columns)

    if model_type == 'default':
        model = MobileNetV2(input_shape=((2, 80, 100),), classes=3, )

    elif model_type == 'descr':
        model =MobileNetV2(input_shape=((2, 80, 100),), classes=2, )

    elif model_type == 'dann':
        model = MobileNetV2_DANN(input_shape=((2, 80, 100),), classifier_classes=3, descriminator_classes = 2)

    model.load_weights('/home/ishokar/march_test/output_weights' + weights_file)

    steps = 0
    for data, row_0 in islice(test_generator(1, 1, df_test, data, model_type), steps_per_epoch):
        df_row.loc[steps] = row_0

        data = data.reshape((1, 2, 80, 100))
        probabilities_0 = model.predict(data, steps = 1)
        probabilities.append(probabilities_0)

        intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[-3].output)
        layer_nodes_0 = intermediate_layer_model.predict(data)
        layer_nodes.append(layer_nodes_0)
        
        if model_type == 'default':
            lab = df_row.loc[steps]['label']
        else:
            lab = df_row.loc[steps]['label']
        print(steps,'/', steps_per_epoch, ':', round((steps*100)/steps_per_epoch, 2), '%,', probabilities_0[0], lab) 
        steps+=1
    
    pkl.dump(layer_nodes, open('files_new/nodes_values_{}_{}.pkl'.format(name, weights_file[8:-3]),'wb'))
    pkl.dump(probabilities, open('files_new/test_probabilities_{}_{}.pkl'.format(name, weights_file[8:-3]),'wb'))
    pkl.dump(df_row, open('files_new/test_df_{}_{}.pkl'.format(name, weights_file[8:-3]),'wb'))
	
    df2 = index_finder(probabilities, df_row)
    pkl.dump(df2, open('df_physics.pkl','wb'))

