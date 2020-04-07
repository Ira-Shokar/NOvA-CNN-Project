from functions import *
from methods import *
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_history(path, file):
    
    with open(path + file,'rb') as f1:
        history = pkl.load(f1)
        
    fig = plt.figure(figsize=(16,8))

    ax1 = fig.add_axes([0, 0, 1, 1])
    ax2 = fig.add_axes()
    ax2 = ax1.twinx() 

    lns1 = ax1.plot(history.history['loss'][:100], color='red', label='loss')
    lns2 = ax1.plot(history.history['val_loss'][:100], color='green', label='val_loss')

    lns3 = ax2.plot(history.history['accuracy'][:100], color='blue', label='accuracy')
    lns4 = ax2.plot(history.history['val_accuracy'][:100], color='orange', label='val_accuracy')

    leg = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc='upper left')
    plt.title('', fontsize=20)

    plt.rcParams.update({'font.size': 14})

    ax1.set_ylim(0.0, 5)
    ax1.set_ylabel('loss')

    ax2.set_ylim(0.1, 0.85)
    ax2.set_ylabel('accuracy')

    ax1.set_xlabel('epochs')

    plt.show()
    
    

def test_results(path, model_name):
    
    with open(path+ 'test_probabilities__{.format(model_name)}.pkl'.format(model_name),'rb') as f1:
        probabilities = pkl.load(f1)
    with open(path+ 'test_df__{}.pkl'.format(model_name),'rb') as f2:
        df = pkl.load(f2)
    with open(path+ 'df_physics_{}.pkl.format(model_name)','rb') as f3:
        physics_df = pkl.load(f3)
    with open(path+ 'nodes_values_default_{}.pkl'.format(model_name),'rb') as f4:
    node_values = pkl.load(f4)
    
    return probabilities, df, physics_df, node_values



def unpack_df():
    labels = list(df['label'])
    gibuu_weights = list(df['weight'])

    events = np.zeros((len(df['file']), 2))
    
    for i in range(len(df['file'])):
        if '_genie_' in str(df['file'][i]):
            events[i] = [1, 0]
        elif 'gibuu' in str(df['file'][i]):
            events[i] = [0, 1]
            
    def sub(x):
        return x-1
    
    test_vals = list(map(sub, labels))   
    
    return labels, gibuu_weights, events, test_vals



def predictions():
    
    predictions = []
    for i in probabilities:
        nc = i[0][0]
        nu_e = i[0][1]
        nu_mu = i[0][2]
        if nc>= nu_e and nc>=nu_mu:
            predictions.append(0)
        elif nu_e>= nc and nu_e>=nu_mu:
            predictions.append(1)
        elif nu_mu>= nu_e and nu_mu>=nc:
            predictions.append(2)

    #accuracy
    acc = 0
    for i in range(len(probabilities)):
        if test_vals[i]==predictions[i]:
            acc+=1
        else:
            pass
    acc/=len(test_vals)
    
    
    print('Accuracy:{} \n'.format(acc))

    #printing first 10 events to check data
    
    print('Probabilities: \n')
    for i in range(10):
        print(probabilities[i], '\n')
    print('Predictions: \n')
    print(predictions[:10], '\n')
    print('Truth labels: \n')
    print(test_vals[:10])
    
    return predictions



def event_hist(data, data_type):
    # data_type options: test_vals, predictions
    
    plt.figure(figsize=(12,6))
    plt.hist(data)
    x = [0.1, 1.1, 1.9]
    class_names = ['nc', 'nu_e', 'nu_mu']
    plt.xticks(x, class_names)
    plt.ylabel('Count')
    if data_type== 'test_vals':
        plt.title('MC Truth Classes')
    elif data_type== 'predictions':
        plt.title('Predicited Classes')
    
    
        
def classifier_output(probabilities, interaction) :      

    if interaction == 'nc':
        index = 0
    elif interaction == 'nu_e':
        index = 1
    elif interaction == 'nu_mu':
        index = 2
        
    mu_e = []
    nc = []
    nu_mu = []
    
    for i in range(len(probabilities)):
        if test_vals[i] ==0 :
            nc.append(probabilities[i][0][index])
        elif test_vals[i] ==1:
            mu_e.append(probabilities[i][0][index])
        elif test_vals[i] ==2:
            nu_mu.append(probabilities[i][0][index])

    plt.figure(figsize=(25,10))
    factor = 1/(len(test_vals))
    
    (counts, bins) = np.histogram(mu_e, bins=100)
    plt.hist(bins[:-1], bins, weights=factor*counts, histtype='step', fill=False, linestyle=('solid'),color=('orange'))

    (counts, bins) = np.histogram(nc, bins=100)
    plt.hist(bins[:-1], bins, weights=factor*counts, histtype='step', fill=False, linestyle=('solid'),color=('b'))

    (counts, bins) = np.histogram(nu_mu, bins=100)
    plt.hist(bins[:-1], bins, weights=factor*counts, histtype='step', fill=False, linestyle=('solid'),color=('g'))

    plt.legend(['nc', 'nu_e','nu_mu'], loc='upper left')
    plt.ylabel('Percentage of test events')
    plt.xlim(0,1)
    plt.xlabel('{} classifer Output'.format(interaction))
    
    
    

def purity_efficiency(probabilites, interaction):
    
    if interaction == 'nc':
        index = 0
    elif interaction == 'nu_e':
        index = 1
    elif interaction == 'nu_mu':
        index = 2
    
    purity_list = []
    efficiency_list = []
    p_x_e_list = []

    for j in np.arange(0, 0.99, 0.01):
        nu_mu_above = []
        nu_mu_below = []
        nc_above = []
        nc_below = []
        nu_e_above = []
        nu_e_below = []
        

        for i in range(len(probabilities)):
            if test_vals[i] ==2:
                if probabilities[i][0][index]>=j:
                    nu_mu_above.append(probabilities[i][0][index]*gibuu_weights[i])
                elif probabilities[i][0][index]<=j:
                    nu_mu_below.append(probabilities[i][0][index]*gibuu_weights[i])

            elif test_vals[i] ==0:         
                if probabilities[i][0][index]>=j:
                    nc_above.append(probabilities[i][0][index]*gibuu_weights[i])
                elif probabilities[i][0][index]<=j:
                    nc_below.append(probabilities[i][0][index]*gibuu_weights[i])

            elif test_vals[i] ==1:        
                if probabilities[i][0][index]>=j:
                    nu_e_above.append(probabilities[i][0][index]*gibuu_weights[i])
                elif probabilities[i][0][index]<=j:
                    nu_e_below.append(probabilities[i][0][index]*gibuu_weights[i])
        
        if interaction == 'nc':
            purity = len(nc_above)/(len(nc_above)+len(nu_mu_above)+len(nu_e_above))
            efficiency = len(nc_above)/(len(nc_above)+len(nc_below))
            
        elif interaction == 'nu_e':
            purity = len(nu_e_above)/(len(nc_above)+len(nu_mu_above)+len(nu_e_above))
            efficiency = len(nu_e_above)/(len(nu_e_above)+len(nu_e_below))
            
        elif interaction == 'nu_mu':      
            purity = len(nu_mu_above)/(len(nc_above)+len(nu_mu_above)+len(nu_e_above))
            efficiency = len(nu_mu_above)/(len(nu_mu_above)+len(nu_mu_below))
            
        purity_list.append(purity*100)
        efficiency_list.append(efficiency*100)
        p_x_e_list.append(purity*efficiency*100)
    
    plt.figure(figsize=(25,10))
    plt.plot(purity_list)
    plt.plot(efficiency_list)
    plt.plot(p_x_e_list)
    plt.xlabel('{} classifer Output Percentage'.format(interaction))
    plt.ylabel('Percentage')
    plt.legend(['Purity', 'Efficiency', 'Purity* Efficiency'], loc='lower left')
    
    
    
    
def roc(probabilities, test_vals):
    pr_nc = []
    pr_nu_e = []
    pr_nu_mu = []
    for i in range(len(probabilities)):
        pr_nc.append(probabilities[i][0][0])
        pr_nu_e.append(probabilities[i][0][1])
        pr_nu_mu.append(probabilities[i][0][2])

    nc_fpr, nc_tpr, nc_thresholds = metrics.roc_curve(test_vals, pr_nc, pos_label=0)
    nu_e_fpr, nu_e_tpr, nu_e_thresholds = metrics.roc_curve(test_vals, pr_nu_e, pos_label=1)
    nu_mu_fpr, nu_mu_tpr, nu_mu_thresholds = metrics.roc_curve(test_vals, pr_nu_mu, pos_label=2)

    plt.figure(figsize=(10,10))
    plt.plot(nc_fpr, nc_tpr, label = 'NC')
    plt.plot(nu_e_fpr, nu_e_tpr, label = 'Nu E')
    plt.plot(nu_mu_fpr, nu_mu_tpr, label = 'Nu Mu')
    plt.legend(['NC', 'Nu E', 'Nu Mu',], loc='upper left')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.plot([0,1], [0,1], '--')
    
    
    
def node_event(node):
    bins = 20
    nc_genie = []
    nc_gibuu = []
    nu_e_genie = []
    nu_e_gibuu = []
    nu_mu_genie = []
    nu_mu_gibuu = []
    for i in range(len(events)):
        if test_vals[i] == 0 and events[i][0] == 1:
            nc_genie.append(node_values[i][0][node])
        elif test_vals[i] == 0 and events[i][1]  == 1:
            nc_gibuu.append(node_values[i][0][node])
        elif test_vals[i] == 1 and events[i][0]  == 1:
            nu_e_genie.append(node_values[i][0][node])
        elif test_vals[i] == 1 and events[i][1]  == 1:
            nu_e_gibuu.append(node_values[i][0][node])
        elif test_vals[i] == 2 and events[i][0]  == 1:
            nu_mu_genie.append(node_values[i][0][node])
        elif test_vals[i] == 2 and events[i][1]  == 1:
            nu_mu_gibuu.append(node_values[i][0][node])

    dataset_ = [nc_genie, nc_gibuu, nu_e_genie, nu_e_gibuu, nu_mu_genie, nu_mu_gibuu]
    label = ['nc_genie', 'nc_gibuu', 'nu_e_genie', 'nu_e_gibuu', 'nu_mu_genie', 'nu_mu_gibuu']

    plt.figure(figsize=(8,5))
    (counts, bins) = np.histogram(nc_genie, bins=bins)
    factor = 1/(len(nc_genie))
    plt.hist(bins[:-1], bins, weights=factor*counts, histtype='step', fill=False, label='nc_genie', linestyle=('solid'),color=('g'))
    
    (counts, bins) = np.histogram(nc_gibuu, bins=bins)
    factor = 1/(len(nc_gibuu))
    plt.hist(bins[:-1], bins, weights=factor*counts, histtype='step', fill=False, label='nc_gibuu', linestyle=('dashed'),color=('g'))
    
    (counts, bins) = np.histogram(nu_e_genie, bins=bins)
    factor = 1/(len(nu_e_genie))
    plt.hist(bins[:-1], bins, weights=factor*counts, histtype='step', fill=False, label='nu_e_genie', linestyle=('solid'),color=('r'))
    
    (counts, bins) = np.histogram(nu_e_gibuu, bins=bins)
    factor = 1/(len(nu_e_gibuu))
    plt.hist(bins[:-1], bins, weights=factor*counts, histtype='step', fill=False, label='nu_e_gibuu', linestyle=('dashed'),color=('r'))
    
    (counts, bins) = np.histogram(nu_mu_genie, bins=bins)
    factor = 1/(len(nu_mu_genie))
    plt.hist(bins[:-1], bins, weights=factor*counts, histtype='step', fill=False, label='nu_mu_genie', linestyle=('solid'),color=('b'))
    
    (counts, bins) = np.histogram(nu_mu_gibuu, bins=bins)
    factor = 1/(len(nu_mu_gibuu))
    plt.hist(bins[:-1], bins, weights=factor*counts, histtype='step', fill=False, label='nu_mu_gibuu', linestyle=('dashed'),color=('b'))

    plt.title('Node Number {}'.format(node))
    plt.legend(prop={'size': 10})
    plt.xlabel('Node Value')
    plt.ylabel('Percentage of Data')
    plt.xlim(0.05,1)
    plt.ylim(0,0.2)
    
    
    
def node_pe(node):
    for j in np.arange(0, 0.25, 0.01):
        purity_list = []
        efficiency_list = []
        p_x_e_list = []
        
        nu_mu_above = []
        nu_mu_below = []
        nc_above = []
        nc_below = []
        nu_e_above = []
        nu_e_below = []
        for i in range(len(probabilities)):
            if test_vals[i] ==2:
                if probabilities[i][0][1]>=j:
                    nu_mu_above.append(node_values[i][0][node])
                elif probabilities[i][0][1]<=j:
                    nu_mu_below.append(node_values[i][0][node])

            elif test_vals[i] ==0:         
                if probabilities[i][0][1]>=j:
                    nc_above.append(node_values[i][0][node])
                elif probabilities[i][0][1]<=j:
                    nc_below.append(node_values[i][0][node])

            elif test_vals[i] ==1:        
                if probabilities[i][0][1]>=j:
                    nu_e_above.append(node_values[i][0][node])
                elif probabilities[i][0][1]<=j:
                    nu_e_below.append(node_values[i][0][node])
                    
                    
    purity = len(nu_e_above)/(len(nc_above)+len(nu_mu_above)+len(nu_e_above))
    purity_list.append(purity)

    efficiency = len(nu_e_above)/(len(nu_e_above)+len(nu_e_below))
    efficiency_list.append(efficiency)

    p_x_e_list.append(purity*efficiency)

    fig = plt.figure(figsize=(20,10))
    plt.plot(purity_list)
    plt.plot(efficiency_list)
    plt.plot(p_x_e_list)
    plt.xlabel('Nu Mu classifer Output')
    plt.ylabel('Percentage')
    plt.title('Trained and Tested on Both Datasets')
    plt.legend(['Purity', 'Efficiency', 'Purity* Efficiency'], loc='lower left')
    
def domain_physics(text, bins):

    plt.figure(figsize=(18,10))
    (counts, bins) = np.histogram(df2[text], bins=bins)
    factor = 1/(len(df2[text]))
    plt.hist(bins[:-1], bins, weights=factor*counts, histtype='step', fill=False, label='Gibuu<0.2')

    (counts, bins) = np.histogram(df3[text], bins=bins)
    factor = 1/(len(df3[text]))
    plt.hist(bins[:-1], bins, weights=factor*counts, histtype='step', fill=False, label= 'Genie<0.2')

    (counts, bins) = np.histogram(df4[text], bins=bins)
    factor = 1/(len(df4[text]))
    plt.hist(bins[:-1], bins, weights=factor*counts, histtype='step', fill=False, label='0.4<Gibuu<0.7')
    
    plt.legend(prop={'size': 14})
    plt.ylabel('Percentage of Data')
    plt.xlabel(text)
    plt.show()
