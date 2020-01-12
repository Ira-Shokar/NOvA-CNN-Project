import os
import pickle as pkl
import h5py
import numpy as np
import copy
import pandas as pd
import random
import mobilenetv2
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data_utils_user import *



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
    arr_str = ('{},{},{},{},{}'.format(run, subrun, evt, subevt, cycle))
    return arr_str



def get_files(path):
    """get_files
    Function creates a list of all the hdf5 files in a directorty
    # Arguments
        path: the path to the direectory
    # Returns
        filenames: list of filenames"""
    filenames = []
    folder = os.fsencode(path)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(('.h5')):
            filenames.append(filename)
            
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

    train_event = []
    for i in range(len(cvnmaps['evt'])):
        train_event.append(encode_event(cvnmaps, i))                   

    mc_event = []
    for i in range(len(mc['evt'])):
        mc_event.append(encode_event(mc, i))

    #mapping
    Train_Params = {}
    for i in range(len(mc_event)):
        for j in range(len(train_event)):
            if mc_event[i] == train_event[j]:
                Train_Params[mc_event[i]] = j, mc['iscc'][i][0], mc['pdg'][i][0]
                        
    return f, file, Train_Params
                
    
    
def mu_cuts(f, file):
    """
    Function that takes in an hdf5 file and creates a list of events that do not pass the nu mu cuts:
    
    nu mu cuts defined - "CAFAna/Cuts/NumuCuts2018.h"
        cuts used - kNumuQuality && kNumuContainFD2017
    """
    cut_arr_mu = []

    ### kNumuQuality cut ###
    en_numu = f['rec.energy.numu']
    for i in range(len(en_numu['trkccE'])):
        if en_numu['trkccE'][i]<=0:
            s = encode_event(en_numu, i)
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
        
        dataframes.append(pd.DataFrame({'run' : run, 
                                        'subrun' : subrun, 
                                        'evt' : evt,
                                        'subevt' : subevt,
                                        'cycle' : cycle,
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

    

def generator(batch_size, dataset, steps_per_epoch):
    batch_images = np.zeros((batch_size, 2, 80, 100))
    batch_labels = np.zeros((batch_size, 3))
    index_list = []
    calls =0
    size = dataset.shape[0]
    while True:
        for i in range(batch_size):
            index= random.randint(0,size-1)
            while index in index_list:
                index= random.randint(0,size-1)
            index_list.append(index)
            calls+=1

            try:
                row = dataset.loc[index]
                file= row['file']
                images  = image(maps(file)[row['train_index']])
                images= (images - np.min(images))/ (np.max(images) - np.min(images))
                labels = np.zeros(3)
                if row['label']==1:
                    labels[0]=1
                elif row['label']==2:
                    labels[1]=1  
                elif row['label']==3:
                    labels[2]=1

            except mp.TimeoutError:
                i-=1
                pass
            if calls%steps_per_epoch ==0:
                index_list = []
  
            batch_images[i] = images
            batch_labels[i] = labels
        
        yield batch_images, batch_labels  

        
        
def open_df(num):
    with open(path+ 'df_{}.pkl'.format(num),'rb') as file:
        df = pkl.load(file)
    return df



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
        
        
        
def test_labels(dataset, index_list):
    labels_list = np.empty((0, 3))
    for i in index_list:
        row = dataset.loc[i]
        labels = np.zeros(3)
        if row['label']==1:
            labels[0]=1
        elif row['label']==2:
            labels[1]=1
        elif row['label']==3:
            labels[2]=1
        labels_list = np.concatenate((labels_list, labels.reshape(1,3)), axis = 0)

    return labels_list
