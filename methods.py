from functions import *
from mobilenetv2 import *
from keras.models import Model

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




def train_mobilenetv2(df_train,
                      df_val,
                      data = 'both',
                      train_type = 'default',
                      epochs=50,
                      dataset_percent = 0.01,
                      batch_size = 32,
                      call_back_patience = 10,
                      learning_rate = 0.0001,
                      model_optimiser = 'Adam',
                      out_file_name = "name"):

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

    steps_per_epoch = round(len(df_train['file'])/(batch_size)*dataset_percent)
    val_steps_per_epoch = round(len(df_val['file'])/(batch_size)*dataset_percent)

    callbacks = [EarlyStopping(monitor='accuracy', patience=call_back_patience)]

    if model_optimiser == 'SGD':
        opt= SGD(lr=learning_rate, momentum=0.9)
    elif model_optimiser == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    else:
	print('Not valid model_optimiser name')

    if train_type == 'dann':
        mobnetmodel = MobileNetV2_DANN(input_shape=((2, 80, 100),), classifier_classes=3, descriminator_classes = 2)
        mobnetmodel.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

        history = mobnetmodel.fit_generator(generator=dann_generator(batch_size, steps_per_epoch, df_train),
                                            steps_per_epoch= steps_per_epoch,
                                            validation_data= dann_generator(batch_size, val_steps_per_epoch, df_val, val='val'),
                                            validation_steps= val_steps_per_epoch,
                                            epochs=epochs, callbacks = callbacks)
    elif train_type== 'default':
        mobnetmodel = mobilenetv2.MobileNetV2(input_shape=((2, 80, 100),), classes=3)
        mobnetmodel.compile(optimizer=opt,
                            loss='categorical_crossentropy',metrics=['accuracy'])

        history = mobnetmodel.fit_generator(generator=generator(batch_size, steps_per_epoch, df_train, data = 'data'),
                                            steps_per_epoch= steps_per_epoch,
                                            validation_data= generator(batch_size, val_steps_per_epoch, df_val, val='val', data = 'data'),
                                            validation_steps= val_steps_per_epoch,
                                            epochs=epochs, callbacks = callbacks)

    elif train_type == 'descr':
        mobnetmodel = mobilenetv2.MobileNetV2(input_shape=((2, 80, 100),), classes=2)
        mobnetmodel.compile(optimizer=opt,
                            loss='categorical_crossentropy',metrics=['accuracy'])

        history = mobnetmodel.fit_generator(generator=descriminator_generator(batch_size, steps_per_epoch, df_train),
                                            steps_per_epoch= steps_per_epoch,
                                            validation_data= descriminator_generator(batch_size, val_steps_per_epoch, df_val, val='val'),
                                            validation_steps= val_steps_per_epoch,
                                            epochs=epochs, callbacks = callbacks)


    mobnetmodel.save_weights("weights_{}.h5".format(out_file_name))
    history.model = None
    pkl.dump(history, open( "history_{}.pkl".format(out_file_name), "wb" ))

    return mobnetmodel, history



def train(path,
          train_type = 'both',
          epochs= 50,
          batch_size = 32,
          dataset_percent = 0.01,
          call_back_patience = 20,
          learning_rate = 0.0001,
          model_optimiser='SGD',
          out_file_name = '32_SGD'):

    df1 = open_df(path, 1)
    df2 = open_df(path, 2)
    df3 = open_df(path, 3)
    df4 = open_df(path, 4)
    df5 = open_df(path, 5)
    df6 = open_df(path, 6)
    df9 = open_df(path, 9)
    df10 = open_df(path, 10)
    df11 = open_df(path, 11)

    df2_ = open_df_gibuu(path, 2)
    df3_ = open_df_gibuu(path, 3)


    if train_type=='both' or train_type == 'dann' or train_type == 'descr':
        train = [df1, df2, df3, df4, df5, df6, df2_]
        val= [df9, df10, df11, df3_]

    elif train_type=='genie':
        train = [df1, df2, df3, df4, df5, df6]
        val= [df9, df10, df11]

    elif train_type=='gibuu':
        df7_ = open_df_gibuu(path, 7)
        df5_ = open_df_gibuu(path, 5)
        df6_ = open_df_gibuu(path, 6)
        train = [df2_, df3_, df7_]
        val= [df5_, df6_]

    df_train = pd.concat(train)
    df_train.index = range(len(df_train['file']))
    df_val = pd.concat(val)
    df_val.index = range(len(df_val['file']))

    if train_type=='both' or train_type=='genie' or train_type=='gibuu':
        model = 'default'
    else:
	model = train_type
    model, history = train_mobilenetv2(df_train,
                                       df_val,
                                       train_type = model,
                                       epochs= epochs,
                                       dataset_percent = dataset_percent,
                                       batch_size = batch_size,
                                       call_back_patience = call_back_patience,
                                       learning_rate = learning_rate,
                                       model_optimiser = model_optimiser,
                                       out_file_name = out_file_name)



def test(weights_file, path, name, data = 'both', model_type = 'default', output = 'default'):

    index_list = []
    test_labels_list =[]
    weight_index = []
    event_list = []

    probabilities  =[]
    layer_nodes = []

    batch_no = 32

    if data =='both':
        df7 = open_df(path, 7)
        df8 = open_df(path, 8)

        df14_ = open_df_gibuu(path, 14)

        test = [df7, df8, df14_]

    elif data =='genie':
        df7 = open_df(path, 7)
        df8 = open_df(path, 8)
        test = [df7, df8]

    elif data =='gibuu':
        df14_ = open_df_gibuu(path, 14)
        df15_ = open_df_gibuu(path, 15)
        test = [df14_, df15_]
    elif data =='genie':
        df7 = open_df(path, 7)
        df8 = open_df(path, 8)
        test = [df7, df8]

    elif data =='gibuu':
        df14_ = open_df_gibuu(path, 14)
        df15_ = open_df_gibuu(path, 15)
        test = [df14_, df15_]


    df_test = pd.concat(test)
    df_test.index = range(len(df_test['file']))

    steps_per_epoch = round(len(df_test['file'])/(batch_no+1))*0.01



    if model_type == 'default':
        model = MobileNetV2(input_shape=((2, 80, 100),), classes=3, )

    elif model_type == 'descr':
        model =MobileNetV2(input_shape=((2, 80, 100),), classes=2, )

    elif model_type == 'dann':
        model = MobileNetV2_DANN(input_shape=((2, 80, 100),), classifier_classes=3, descriminator_classes = 2)

    model.load_weights('/home/ishokar/feb_test/output_weights' + weights_file)

    steps = 0
    for data, test_labels_list_0, weight_index_0, event_list_0 in test_generator(batch_no, steps_per_epoch, df_test, data, model_type):
        steps+=1
        if output == 'nodes':
            intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[-3].output)
            layer_nodes_0 = intermediate_layer_model.predict(data)

        else:
            probabilities_0 = model.predict(data, steps = batch_no)

        for i in range(batch_no):
            test_labels_list.append(test_labels_list_0[i])
            weight_index.append(weight_index_0[i])
            event_list.append(event_list_0[i])

            if output == 'nodes':
               layer_nodes.append(layer_nodes_0[i])
            else:
               probabilities.append(probabilities_0[i])

        print(round((steps*100)/steps_per_epoch, 2), '%')
        if steps==steps_per_epoch:
            break

    if output == 'nodes':
        with open('files/nodes_values_{}.pkl'.format(i[8:-3]),'wb') as f1:
            pkl.dump(layer_nodes, f1)

    else:
	with open('files/test_probabilities{}_{}.pkl'.format(name, weights_file[8:-3]),'wb') as f1:
            pkl.dump(probabilities, f1)

    with open('files/nodes_events_{}.pkl'.format(weights_file[8:-3]),'wb') as f2:
        pkl.dump(event_list, f2)

    with open('files/test_labels_{}_{}.pkl'.format(name, weights_file[8:-3]),'wb') as f3:
        pkl.dump(test_labels_list, f3)

    with open('files/test_weights_list_short_{}_{}.pkl'.format(name, weights_file[8:-3]),'wb') as f4:
        pkl.dump(weight_index, f4)
