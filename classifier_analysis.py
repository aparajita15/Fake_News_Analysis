from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np
import time
start_time = time.time();
import random
import pdb
import sys
import csv
import pickle
import os 
import argparse
import random

### Classifier and accuracy/auc analysis


def get_acc_auc(pred, actual):
    actual =np.array(actual)
    # Checking accuracy only for the values where the value of label is 0 or 1
    indices = np.where(actual>=0)
    ac = actual[indices]    
    pd = pred[indices]
    auc = roc_auc_score(ac, pd)
    #fpr, tpr, thres = (roc_curve(ac, pd, pos_label=2)) 
    accuracy = sum( ac ==pd ) * 1.0 / len(ac)

    print('\n Accuracy obtained on the test set: ' + str(accuracy ))
    print('\n AUC obtained on the test set: ' + str(auc ))
   
    return accuracy, auc



def get_split(data, labels, percent):

            list_1 = np.arange(len(data)) # example list
            np.random.shuffle(list_1)
            
            ind = int(len(data)*percent)
            pdb.set_trace()
            sec1 = data[list_1[:ind]]
            lab1 = labels[list_1[:ind]]
            sec2 = data[list_1[ind:]]
            lab2 = labels[list_1[ind:]]
            

            return sec1, lab1, sec2, lab2
'''
def get_split(data, labels, percent):
            sec1 = data.sample(frac=percent)
            lab1 = labels.loc[data.index.isin(sec1.index)]
            sec2 = data.loc[~data.index.isin(sec1.index)]
            lab2 = labels.loc[~data.index.isin(sec1.index)]
            
            return sec1, lab1, sec2, lab2
'''
def timer1(end):
    global start_time
    start = start_time
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

                    
def main(label='mediabiasfactcheck_mediabiasfactcheck', clf_opt='log_reg'):

    path = 'C:/Users/Aparajita/Desktop/FakeNews/Pickled_data'

    ##############Loading the training and test dataset
    with open('Pickled_data/training_file_everything_without', 'rb') as wr:
        train_set, train_stances, label_names = pickle.load( wr)
    with open('Pickled_data/testing_file_everything_without', 'rb') as wr:
        test_set, actual_labels, label_names = pickle.load( wr)

    #pdb.set_trace()
    
    reals  = ['mediabiasfactcheck', 'alexa', 'vargo']
    fakes =['mediabiasfactcheck', 'politifact', 'zimdars', 'min2', 'dailydot']
    classifiers  = ['log_reg', 'svm', 'dec_tree', 'mlp']

    for ir in reals:
        for ifa in fakes:
                for clf_opt in classifiers:
    
                    label = ir+'_'+ifa

                    ##Prepare the data accordingly
                    label_index = label_names.index(label)
                    y_train_label = []
                    y_test_label = []
                    for i in range(len(train_stances)):
                        y_train_label+= [ train_stances[i][label_index]  ]

                    for i in range(len(actual_labels)):                            
                            y_test_label+= [ actual_labels[i][label_index]  ]
                    
                    ###########################Splitting into training and vlaidation test
                    ###############Balancing the training dataset
                    #val_perc = 0.8
                    #train_set, train_label , vali_set, val_label = get_split(train_set, train_stances, val_perc)
                    real_indices  = [i for i, x in enumerate(y_train_label) if x == 1]  
                    fake_indices  = [i for i, x in enumerate(y_train_label) if x == 0]  

                    mn = min(len(real_indices), len(fake_indices))
                    t_real = [train_set[i] for i in real_indices]
                    t_fake = [train_set[i] for i in fake_indices]
                    #Shuffling needed
                    t_x = t_real[:mn] + t_fake[:mn]
                    t_y = [1]*mn+ [0]*mn

                    
                    '''
                    if ratio<1:
                                    fake_sample = fake_sample.sample(frac=ratio)
                    else:
                                    real_sample = real_sample.sample(frac=1/ratio)

                    print('Downsampling ratio ' + str(ratio))
                    train_percent = 0.8


                    real_s1 , real_s2 = get_split(real_sample, train_percent)
                    fake_s1 , fake_s2 = get_split(fake_sample, train_percent)

                    ## The balanced sampled train and test set
                    train = pd.concat([real_s1, fake_s1])
                    test = pd.concat([real_s2, fake_s2])
                    '''
                    ####################################

                    if clf_opt == 'log_reg':
                        ### Logistic regression classifier begins:i
                        print('Logistic Regression classifier started')
                        #timer1(end = time.time())
                        param_grid = {'C': [ 0.1, 1 ] ,'penalty': [ 'l2'] , 'solver' : [ 'liblinear', 'sag']  }
                        clf = GridSearchCV(LogisticRegression( max_iter=10000  , class_weight ='balanced',verbose = True,  random_state=0, solver='lbfgs', n_jobs = 4  ), n_jobs=4 ,scoring='roc_auc', param_grid = param_grid ,verbose = 100 , cv=StratifiedKFold()).fit(t_x, t_y)
                        #clf = LogisticRegression(class_weight ='balanced' , dual = True, solver = 'liblinear',  random_state=0 ).fit(train_set, train_stances)   
                        #clf = LogisticRegression(class_weight = {1: train_ratio/(train_ratio+1), 0: 1/(train_ratio + 1)}, random_state=0, solver='lbfgs',multi_class='multinomial').fit(train_set, train_stances)
                        print('\n Training completed')

                    elif clf_opt == 'svm':
                        print('SVM classifier started')

                        C_range = np.logspace(-2, 10, 13)
                        gamma_range = np.logspace(-9, 3, 13)
                        param_grid = dict(gamma=gamma_range, C=C_range)
                        cv =StratifiedKFold()#StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
                        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv,  n_jobs=4, scoring='roc_auc' )
                        grid.fit(train_set, train_stances)

                        #timer1(end = time.time())
                        clf = grid
                        print('\n Training completed')


                    elif clf_opt == 'dec_tree':
                        print('Decision Tree classifier started')
                        timer1(end = time.time())
                        #clf =  DecisionTreeClassifier(class_weight = {1: train_ratio/(train_ratio+1), 0: 1/(train_ratio + 1)}   ).fit(train_set, train_stances)
                        
                        rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 
                        param_grid = {  'n_estimators': [200, 700],  'max_features': ['auto', 'sqrt', 'log2'] }
                        CV_rfc = GridSearchCV(estimator=rfc,param_grid=param_grid, cv=StratifiedKFold(), n_jobs = 4,  scoring='roc_auc')
                        CV_rfc.fit(train_set, train_stances)
                        clf = CV_rfc
                        
                        print('\n Training completed')


                    elif clf_opt=='mlp':
                        ### Logistic regression classifier begins:i
                        print('MLP classifier started')
                        timer1(end = time.time())
                        clf = MLPClassifier(learning_rate_init = 0.005, hidden_layer_sizes=(2000,),alpha=0.0001, verbose=True, early_stopping=True).fit(train_set, train_stances)
                        print('\n Training completed')

                    

                    ###########################################################################
                    ###########################################################################
                    '''   Accurracy AUC Analysis'''
                    ###########################################################################
                    f6 = 'C:/Users/Aparajita/Desktop/FakeNews/Saved_classifiers/'+ clf_opt +'_trained_on'+label +'_labels'
                    with open(f6, 'wb') as f:
                        pickle.dump(clf, f)

                    test_stances = clf.predict(test_set) 
                    acc, auc = get_acc_auc(test_stances, y_test_label)

    
if __name__== '__main__':
    main()  
    ## Classifiers saved
    ## Accuracy and AUC calculated