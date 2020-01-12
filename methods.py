from functions import *



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

        out_path = '/home/ishokar/december_test/output'

        df = pd.concat(dataframes)
        df.index = range(len(df['file']))

        with open(out_path + '/df_{}.pkl'.format(batch_no),'wb') as f1:
            pkl.dump(df, f1)
            
            

def train_mobilenetmodelv2(df_train,
                           df_val,
                           epochs=40,
                           batch_size = 32,
                           call_back_patience = 5,
                           name = "name"):
    
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
                
    steps_per_epoch = round(len(df_train['file'])/(batch_size)-1)
    val_steps_per_epoch = round(len(df_val['file'])/(batch_size)-1)    

    callbacks = [EarlyStopping(monitor='accuracy', patience=call_back_patience)]
      
    mobnetmodel = mobilenetv2.MobileNetV2(input_shape=((2, 80, 100),), classes=3)     
    mobnetmodel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                        loss='categorical_crossentropy',metrics=['accuracy'])
    

    history = mobnetmodel.fit_generator(generator=generator(batch_size, df_train, steps_per_epoch),
                                        steps_per_epoch= steps_per_epoch, 
                                        validation_data= generator(batch_size, df_val, val_steps_per_epoch),
                                        validation_steps= val_steps_per_epoch,
                                        epochs=epochs, callbacks = callbacks)
                
    mobnetmodel.save_weights("weights_{}.h5".format(name))
    pkl.dump(history, open( "history_{}.pkl".format(name), "wb" ) )            
    
    return mobnetmodel, history



def train(path, epochs= 200, batch_size = 32, call_back_patience = 20, name = '32_Adam'):

    df1 = open_df(path, 1)
    df2 = open_df(path, 2)
    df3 = open_df(path, 3)
    df4 = open_df(path, 4)
    df5 = open_df(path, 5)
    df6 = open_df(path, 6)
    df7 = open_df(path, 7)
    df10 = open_df(path, 10)
    df11 = open_df(path, 11)

    train = [df1, df2, df3, df4, df5, df6, df7]
    val = [df10, df11]

    df_train = pd.concat(train)
    df_train.index = range(len(df_train['file']))
    df_val = pd.concat(val)
    df_val.index = range(len(df_val['file']))

    model, history = train_mobilenetmodelv2(df_train,
                                            df_val,
                                            epochs= epochs,
                                            batch_size = batch_size,
                                            call_back_patience = call_back_patience,
                                            name = name)
    
