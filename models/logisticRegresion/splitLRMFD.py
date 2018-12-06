#!/usr/bin/env python3

import logging
import sys, getopt
import pandas as pd
import numpy as np
import pickle
import imp

from sklearn.model_selection import StratifiedShuffleSplit

inputfile = ''
testfile = ''
trainfile = ''
scorefile = ''


def readCsv(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print ("Error: the file", filename, "does not exist.")
        sys.exit(2)    
    except getopt.GetoptError as e:
        print ("Error:",filename, "error in reading." )
        print (e)     
        sys.exit(2)
    except pd.io.common.EmptyDataError:
        print ("Error: The file",filename,"is empty" )
        sys.exit(2)


def split(tsize,X,Y,trfile,tefile,sfile, nsplits=1,rstate=0):
    
    splitter = StratifiedShuffleSplit(n_splits=nsplits, test_size=tsize, random_state=rstate)
    n,m = X.shape
    
    for train_indices, test_indices in splitter.split(X, Y):        
        X_train, Y_train = X.iloc[train_indices] , Y.iloc[train_indices]
        X_test,  Y_test  = X.iloc[test_indices]  , Y.iloc[test_indices]

        #print(X_test)        
        X_test.to_csv(tefile)
        print('Test file saved in: ' + tefile)
        
        Z_score  = np.column_stack((X_test,Y_test))
        Z_score  = pd.DataFrame(Z_score, columns= ['V%d' % number for number in range(1, m+1)]+["Class"])
        Z_score.to_csv(sfile)
        #print(Z_score)        
        print('Score file saved in: ' + sfile)

        
        Z_train  = np.column_stack((X_train,Y_train))
        Z_train  = pd.DataFrame(Z_train, columns= ['V%d' % number for number in range(1, m+1)]+["Class"])
        Z_train.to_csv(trfile)
        #print(Z_train)
        print('Train file saved in: ' + trfile)

def main(argv):

    if len(sys.argv) <= 10:
        print ('Use: splitLRMFD.py -i <csv_to_split_dataset> -t <csv_train_dataset> -o <csv_test_dataset> -s <csv_score_dataset> -p percent_test_size')
        sys.exit(1)    
    try:
        opts, args = getopt.getopt(argv,"hi:t:o:s:p:",["ifile=","tfile=","ofile=","sfile=","percent"])
    except getopt.GetoptError:
        print ('Use: splitLRMFD.py -i <csv_to_split_dataset> -t <csv_train_dataset> -o <csv_test_dataset> -s <csv_score_dataset> -p percent_test_size')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('Use: splitLRMFD.py -i <csv_to_split_dataset> -t <csv_train_dataset> -o <csv_test_dataset> -s <csv_score_dataset> -p percent_test_size') 
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-t", "--tfile"):
            trainfile = arg            
        elif opt in ("-o", "--ofile"):
            testfile = arg
        elif opt in ("-s", "--sfile"):
            scorefile = arg
        elif opt in ("-p", "--percent"):
            tsize = float(arg)
                

    data = readCsv(inputfile)
    n,m = data.shape

    columns =  ['V%d' % number for number in range(1, m)]    
    target = 'Class'
        
    X = data[columns]
    Y = data[target]
    
    split(tsize,X,Y,trainfile,testfile,scorefile)
            
                    
if __name__ == "__main__":
    main(sys.argv[1:])


        
