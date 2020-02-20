import h5py
import numpy as np
import copy
import pandas as pd
import random
import mobilenetv2
import tensorflow as tf
import keras
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


def encode_event(array, i):
    """encode_event
    Function that takes in an hdf5 array and an index gives the run, subrun, evt, subevt, cycle of an event
    and returnsthem as a string for quick lookups

    # Arguments
        array: branch from the hdf5 files
        i: index within the array
    # Returns
	string containing run, subrun, evt, subevt, cycle
    """
    run = array['run'][i]
    subrun = array['subrun'][i]
    evt = array['evt'][i]
    subevt = array['subevt'][i]
    cycle = array['cycle'][i]
    try:
        genweight = array['genweight'][i]
        arr_str = ('{},{},{},{},{},{}'.format(run, subrun, evt, subevt, cycle, genweight))
    except KeyError:
        arr_str = ('{},{},{},{},{}'.format(run, subrun, evt, subevt, cycle))
    return arr_str



def get_files(path):
    """get_files
    Function creates a list of all the hdf5 files in a directorty
    # Arguments
        path: the path to the direectory
    # Returns
	filenames: list of filenames"""
    folder = os.listdir(os.fsencode(path))
    filenames = [os.fsdecode(file) for file in folder if os.fsdecode(file).endswith(('.h5'))]

    return filenames



def data(file):
    """data
    Function that extracts the data from the hdf5 file and returns the relevant data

    # Arguments
        file: hdf5 file name
    # Returns
	f - the hfd5 file
        file - hfd5 file name
        Train_Params - a dictionary with keys = [run, subrun, evt, subevt, cycle] and
        values = [train array index, iscc value, pdg value]
        maps - the arrays containing the image data
        cvnmaps - the branch of the hdf5 file that contains the maps
    """
    #import file
    f = h5py.File(file, 'r')

    # cvn training data branch and truth branch keys
    cvnmaps = f['rec.training.cvnmaps']
    mc = f['rec.mc.nu']
    train_event = [encode_event(cvnmaps, i) for i in range(len(cvnmaps['evt']))]
    mc_event = [encode_event(mc, i) for i in range(len(mc['evt']))]

    #mapping
    Train_Params = {}
    for i in range(len(mc_event)):
        for j in range(len(train_event)):
            #only including the run, subrun, evt, subevt, cycle data from the mc branch in the matching
            mc_data  = ','.join(mc_event[i].split(',')[:-1])
            train = train_event[j]
            if mc_data == train:
                Train_Params[mc_event[i]] = j, mc['iscc'][i][0], mc['pdg'][i][0]

    return f, file, Train_Params



def mu_cuts(f, file):
    """
    Function that takes in an hdf5 file and creates a list of events that do not pass the nu mu cuts:

    nu mu cuts defined - "CAFAna/Cuts/NumuCuts2018.h"
        cuts used - kNumuQuality && kNumuContainFD2017
    """
    files_list = [os.fsdecode(file) for file in folder if os.fsdecode(file).endswith(('.h5'))]
    cut_arr_mu = []

    ### kNumuQuality cut ###
    en_numu = f['rec.energy.numu']
    for i in range(len(en_numu['trkccE'])):
        if en_numu['trkccE'][i]<=0:
            s = encode_event(en_numu, i)
            if s not in cut_arr_mu:
                cut_arr_mu.append(s)

    cut = [encode_event(sel_remid, i) for i in len(en_numu['trkccE']) if en_numu['trkccE'][i]<=0]
    s = encode_event(cut, i)
    if s not in cut_arr_mu:
	cut_arr_mu.append(s)

    sel_remid = f['rec.sel.remid']
    for i in range(len(sel_remid['pid'])):
        if sel_remid['pid'][i]<=0:
            s = encode_event(sel_remid, i)
            if s not in cut_arr_mu:
                cut_arr_mu.append(s)

    slc = f['rec.slc']
    for i in range(len(slc['nhit'])):
        if slc['nhit'][i]<=20 or slc['ncontplanes'][i]<=4:
            s = encode_event(slc, i)
            if s not in cut_arr_mu:
                cut_arr_mu.append(s)

    cosmic = f['rec.trk.cosmic']
    for i in range(len(cosmic['ntracks'])):
        if cosmic['ntracks'][i]<=0 :
            s = encode_event(cosmic, i)
            if s not in cut_arr_mu:
                cut_arr_mu.append(s)

    ### kNumuContainFD2017 cut ###
    shwlid = f['rec.vtx.elastic.fuzzyk.png.shwlid']
    for i in range(f['rec.vtx.elastic.fuzzyk.png.shwlid']['start.x'].shape[1]):
        a = min(shwlid['start.x'][i],shwlid['stop.x'][i])
        b = max(shwlid['start.x'][i],shwlid['stop.x'][i])
        c = min(shwlid['start.y'][i], shwlid['stop.y'][i])
        d = max(shwlid['start.y'][i], shwlid['stop.y'][i])
        e = min(shwlid['start.z'][i], shwlid['stop.z'][i])
        f = max(shwlid['start.z'][i], shwlid['stop.z'][i])
        if a <=-180 or b >=180 or c <=-180 or d >=180 or e <=20 or f >=1525:
            s = encode_event(shwlid, i)
            if s not in cut_arr_mu:
                cut_arr_mu.append(s)

    f = h5py.File(file, 'r')
    kal_track = f['rec.trk.kalman.tracks']
    for i in range(len(kal_track['start.z'])):
        if kal_track['start.z'][i]>1275 or kal_track['stop.z'][i]>1275:
            s = encode_event(kal_track, i)
            if s not in cut_arr_mu:
                cut_arr_mu.append(s)

    slc = f['rec.slc']
    for i in range(len(slc['firstplane'])):
        if slc['firstplane'][i]<=1 or slc['lastplane'][i]==212 or slc['lastplane'][i]==213:
            s = encode_event(slc, i)
            if s not in cut_arr_mu:
                cut_arr_mu.append(s)

    sel = f['rec.sel.contain']
    for i in range(len(sel['kalfwdcellnd'])):
        if sel['kalyposattrans'][i]>=55 or sel['kalbakcellnd'][i]<10 or sel['kalfwdcellnd'][i]<5:
            s = encode_event(sel, i)
            if s not in cut_arr_mu:
                cut_arr_mu.append(s)

    return cut_arr_mu



def apply_cuts(Train_Params, cut_arr, file):
    """apply_cuts
    Function that the list of cut events and removes them from the mctruth events dictionary

    # Arguments
        Train_Params: the dictionary of events containing iscc and pdg data
        cut_arr: the list of events that did not pass the cuts
    # Returns
	Train_Params_Cut: the new dictionary of events
        events: the training data branch array index values of the events that passed the cut
        train: 2-dimential array, iscc, pdg
        train_val: 1-dimential array, iscc
    """
    Train_Params_Cut= copy.deepcopy(Train_Params)
    for i in cut_arr:
        if i in Train_Params_Cut.keys():
            del Train_Params_Cut[i]

    dataframes = []
    for key in Train_Params_Cut.keys():
        iscc = Train_Params_Cut[key][1]
        pdg = Train_Params_Cut[key][2]
        if iscc == 0:
            interaction = 1
        elif iscc == 1 and pdg*pdg ==144:
            interaction = 2
        elif iscc == 1 and pdg*pdg ==196:
            interaction = 3

        key_split = key.split(',')
        run = key_split[0][1:-1]
        subrun = key_split[1][1:-1]
        evt = key_split[2][1:-1]
        subevt = key_split[3][1:-1]
        cycle = key_split[4][1:-1]
        weight = key_split[5][1:-1]

        dataframes.append(pd.DataFrame({'run' : run,
                                        'subrun' : subrun,
                                        'evt' : evt,
                                        'subevt' : subevt,
                                        'cycle' : cycle,
                                        'weight': weight,
                                        'train_index' :  Train_Params_Cut[key][0],
                                        'label' : interaction,
                                        'file' : [file]
                                       }))

    df = pd.concat(dataframes)

    return df



def e_cuts(f, file):
    """e_cuts
    Function that takes in an hdf5 file and creates a list of events that do not pass the nu e cuts:

    nu mu cuts defined - "CAFAna/Cuts/NueCuts2017.h"
        cuts used - kNue2017NDFiducial && kNue2017NDContain && kNue2017NDFrontPlanes
    """
    ## Nu E Preselection Cuts  #
    cut_arr_e = []

    f = h5py.File(file, 'r')
    # kNue2017NDFiducial
    vtx_el = f[ 'rec.vtx.elastic']
    for i in range(len(vtx_el['vtx.x'])):
        a= vtx_el['vtx.x'][i]
        b= vtx_el['vtx.y'][i]
        c= vtx_el['vtx.z'][i]
        if a<=-100 or a>=160 or b<=-160 or b>=100 or c<=150 or c>=900:
            s = encode_event(vtx_el, i)
            if s not in cut_arr_e:
                cut_arr_e.append(s)

    # kNue2017NDContain
    shwlid = f['rec.vtx.elastic.fuzzyk.png.shwlid']
    for i in range(len(f['rec.vtx.elastic.fuzzyk.png.shwlid']['start.x'])):
        a = min(shwlid['start.x'][i],shwlid['stop.x'][i])
        b= max(shwlid['start.x'][i],shwlid['stop.x'][i])
        c= min(shwlid['start.y'][i], shwlid['stop.y'][i])
        d= max(shwlid['start.y'][i], shwlid['stop.y'][i])
        e = min(shwlid['start.z'][i], shwlid['stop.z'][i])
        f= max(shwlid['start.z'][i], shwlid['stop.z'][i])
        if a <=-170 or b >=170 or c <=-170 or d >=170 or e <=100 or f >=1225:
            s = encode_event(shwlid, i)
            if s not in cut_arr_e:
                cut_arr_e.append(s)
        pass

    # kNue2017NDFrontPlanes
    f = h5py.File(file, 'r')
    sel = f['rec.sel.contain']
    for i in range(len(sel['nplanestofront'])):
        if sel['nplanestofront'][i]<=6:
            s = encode_event(sel, i)
            if s not in cut_arr_e:
                cut_arr_e.append(s)

    return cut_arr_e



def maps(file):
    """maps Function that returns event maps from the hdf5 file
    # Arguments, file: hdf5 file name
    # Returns, maps - the arrays containing the image data
    """
    #import file
    f = h5py.File(file, 'r')

    # cvn training data branch
    cvnmaps = f['rec.training.cvnmaps']
    maps = cvnmaps['cvnmap']

    return maps



def image(maps):
    """image
    Function that the array containing the images and formats them correctly
    # Arguments
        maps: the array containing the images
    # Returns
	image_dataset: nested array containing two images for each event for the z-y and z-x planes
    """
    image_dataset = np.zeros(shape=(2,80,100))
    image_dataset[0,:,:] = np.rot90(np.asarray(maps[:8000]).reshape(100,80))
    image_dataset[1,:,:] = np.rot90(np.asarray(maps[8000:]).reshape(100,80))

    return image_dataset



def event(data, label):
    """event
    Function that takes in an array of event images and displays them

    # Arguments
        data: image array
        i: index within the array
    # Returns
	plot of image along with associated data
    """
    #first view z-x plane
    raw_im_1_cc = data[0,:,:]
    #second view z-y plane
    raw_im_2_cc = data[1,:,:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9))

    #axis lables
    ax1.imshow(raw_im_1_cc, aspect='auto')
    ax1.set_xlabel('z (units)')
    ax1.set_ylabel('x (units)')
    ax2.imshow(raw_im_2_cc, aspect='auto')
    ax2.set_xlabel('z (units)')
    ax2.set_ylabel('y (units)')
    fig.suptitle(label)



def open_df(path, num):
    with open(path+ 'df_{}.pkl'.format(num),'rb') as file:
        df = pkl.load(file)
    return df



def open_df_gibuu(path, num):
    with open(path+ 'df_gibuu_{}.pkl'.format(num), 'rb') as file:
        try:
            df = pkl.load(file)
            return df
        except pkl.UnpicklingError:
            print('Unpickling Error with gibuu df file {}'.format(num))
            pass


def generator(batch_size, steps_per_epoch, dataset, val= 'train', data='both'):

    batch_images = np.zeros((batch_size, 2, 80, 100))
    batch_labels = np.zeros((batch_size, 3))
    steps = 0

    while True:
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        for index in range(len(dataset['file'])):
            row = dataset.loc[index]
            weight = float(row['weight'])

            rand = 20*random.random()
            rand2, rand3, rand4  = random.random(), random.random(), random.random()

            if val == 'train':
                threshold = 0.75
            else:
                threshold = 0.875

            if data != 'both':
                threshold = 0

            if ('gibuu' in str(row['file']) and weight<=rand) or \
               (rand2 <= 0.98 and abs(float(dataset.loc[index]['label'])-3)<=10**(-5)) or \
               (rand3 <= 0.82 and abs(float(dataset.loc[index]['label'])-1)<=10**(-5)) or \
               (rand4 <= threshold and 'gibuu' in str(row['file'])):
                pass

            else:
                images = image(maps(row['file'])[row['train_index']])
                images = (images - np.min(images))/ (np.max(images) - np.min(images))

                if abs(row['label']- 1) <=10**(-5):
                    labels = [1, 0, 0]
                elif abs(row['label']- 2) <=10**(-5):
                    labels = [0, 1, 0]
                elif abs(row['label']- 3) <=10**(-5):
                    labels = [0, 0, 1]

                batch_images[steps%batch_size] = images
                batch_labels[steps%batch_size] = labels

                steps+=1

                if steps%batch_size==0 and steps!=0:
                    yield batch_images, batch_labels

                    

def descriminator_generator(batch_size, steps_per_epoch, dataset, val='train'):

    batch_images = np.zeros((batch_size, 2, 80, 100))
    batch_labels = np.zeros((batch_size, 2))
    steps = 0

    while True:
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        for index in range(len(dataset['file'])):
            row = dataset.loc[index]
            weight = float(row['weight'])

            rand = 20*random.random()
            rand2, rand3, rand4  = random.random(), random.random(), random.random()

            if val == 'train':
                threshold = 0.75
            else:
                threshold = 0.875

            if ('gibuu' in str(row['file']) and weight<=rand) or \
               (rand2 <= 0.98 and abs(float(dataset.loc[index]['label'])-3)<=10**(-5)) or \
               (rand3 <= 0.82 and abs(float(dataset.loc[index]['label'])-1)<=10**(-5)) or \
               (rand4 <= threshold and 'gibuu' in str(row['file'])):
                pass

            else:
                images = image(maps(row['file'])[row['train_index']])
                images = (images - np.min(images))/ (np.max(images) - np.min(images))

                if '_genie_' in str(row['file']):
                    labels = [1, 0]
                elif 'gibuu' in str(row['file']):
                    labels = [0, 1]

                batch_images[steps%batch_size] = images
                batch_labels[steps%batch_size] = labels

                steps+=1

                if steps%batch_size==0 and steps!=0:
                    yield batch_images, batch_labels




def dann_generator(batch_size, steps_per_epoch, dataset, val='train'):

    batch_images = np.zeros((batch_size, 2, 80, 100))
    batch_class_labels = np.zeros((batch_size, 3))
    batch_source_labels = np.zeros((batch_size, 2))
    steps = 0

    while True:
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        for index in range(len(dataset['file'])):
            row = dataset.loc[index]
            weight = float(row['weight'])

            rand = 20*random.random()
            rand2, rand3, rand4  = random.random(), random.random(), random.random()

            if val == 'train':
                threshold = 0.75
            else:
                threshold = 0.875

            if ('gibuu' in str(row['file']) and weight<=rand) or \
               (rand2 <= 0.98 and abs(float(dataset.loc[index]['label'])-3)<=10**(-5)) or \
               (rand3 <= 0.82 and abs(float(dataset.loc[index]['label'])-1)<=10**(-5)) or \
               (rand4 <= threshold and 'gibuu' in str(row['file'])):
                pass

            else:
                images = image(maps(row['file'])[row['train_index']])
                images = (images - np.min(images))/ (np.max(images) - np.min(images))


                if abs(row['label']- 1) <=10**(-5):
                    labels = [1, 0, 0]
                elif abs(row['label']- 2) <=10**(-5):
                    labels = [0, 1, 0]
                elif abs(row['label']- 3) <=10**(-5):
                    labels = [0, 0, 1]

                if '_genie_' in str(row['file']):
                    source = [1, 0]
                elif 'gibuu' in str(row['file']):
                    source = [0, 1]

                batch_images[steps%batch_size] = images
                batch_class_labels[steps%batch_size] = labels
                batch_source_labels[steps%batch_size] = source

                steps+=1

                if steps%batch_size==0 and steps!=0:
                    yield batch_images, [batch_class_labels, batch_source_labels]


def test_generator(batch_size, steps_per_epoch, dataset, data='both', model_type = 'default'):

    batch_images = np.zeros((batch_size, 2, 80, 100))
    labels = []
    weight_index = []
    event_list = []
    steps = 0

    while True:
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        for index in range(len(dataset['file'])):
            row = dataset.loc[index]
            weight = float(row['weight'])
            rand = 20*random.random()
            rand2, rand3, rand4  = random.random(), random.random(), random.random()

            if data != 'both':
                threshold = 0
            else:
                threshold = 0.92

            if ('gibuu' in str(row['file']) and weight<=rand) or \
               (rand2 <= 0.98 and abs(float(dataset.loc[index]['label'])-3)<=10**(-5)) or \
               (rand3 <= 0.82 and abs(float(dataset.loc[index]['label'])-1)<=10**(-5)) or \
               (rand4 <= threshold and 'gibuu' in str(row['file'])):
                pass

            else:
                images = image(maps(row['file'])[row['train_index']])
                images = (images - np.min(images))/ (np.max(images) - np.min(images))

                if '_genie_' in str(row['file']):
                    mc = [1, 0]
                elif 'gibuu' in str(row['file']):
                    mc = [0, 1]

                if abs(row['label']- 1) <=10**(-5):
                    label = [1, 0, 0]
                elif abs(row['label']- 2) <=10**(-5):
                     label = [0, 1, 0]
                elif abs(row['label']- 3) <=10**(-5):
                     label = [0, 0, 1]

                labels.append(label)
                weight_index.append(row['weight'])
                event_list.append(mc)

                steps+=1

                if steps%batch_size==0 and steps!=0:
                    yield batch_images, labels, weight_index, event_list
                    labels = []
                    weight_index = []
                    event_list = []  

			
def equal_data_generator(dataset, method):
    dataframe = pd.DataFrame(columns=dataset.columns)
    steps = 0
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    a = b= c= d= e= 0


    for index in range(len(dataset['file'])):
        row = dataset.loc[index]
        weight = float(row['weight'])
        rand = 10*random.random()
        rand2, rand3, rand4  = random.random(), random.random(), random.random()

        if method == 'train':
            val1 = 0.7529
            val2 = 0.9762
            val3 = 0.8876

        elif method == 'val':
            val1 = 0.8457
            val2 = 0.9762
            val3 = 0.8876
	
	elif method == 'test':
            val1 = 0.6752
            val2 = 0.9767
            val3 = 0.8981

        if ('gibuu' in str(row['file']) and weight<=rand) or \
            (rand4 <= val1 and 'gibuu' in str(row['file'])) or\
            (rand2 <= val2 and abs(float(dataset.loc[index]['label'])-3)<=10**(-5)) or \
            (rand3 <= val3 and abs(float(dataset.loc[index]['label'])-1)<=10**(-5)):
            steps+=1
            pass


        else:
            images = image(maps(row['file'])[row['train_index']])
            images = (images - np.min(images))/ (np.max(images) - np.min(images))

            if abs(row['label']- 1) <=10**(-5):
                a+=1
            elif abs(row['label']- 2) <=10**(-5):
                b+=1
            elif abs(row['label']- 3) <=10**(-5):
                c+=1

            if '_genie_' in str(row['file']):
                d+=1
            elif 'gibuu' in str(row['file']):
                e+=1

            steps+=1
            print('{} %, Nu_mu: {}, Nu_e: {}, NC: {}, Genie: {}, Gibuu: {},'.format(round(step*100/dataset_len, 3), c, b, a, d ,e))

            dataframe.append(row)

    return dataframe
