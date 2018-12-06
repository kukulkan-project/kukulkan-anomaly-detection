#!/usr/bin/env python3

import logging
import sys, getopt
import pandas as pd
import numpy as np
import pickle
import imp
import copy

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


inputfile = ''
outputfile = ''
testfile = ''
scorefile = ''

def isBlank (myString):
    return not (myString and myString.strip())


def normalize(X):
    """
    Make the distribution of the values of each variable similar by subtracting the mean and by dividing by the standard deviation.
    """
    
    for feature in X.columns:
        X[feature] -= X[feature].mean()
        X[feature] /= X[feature].std()
    return X


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
    
def makeAPrediction(model,testfile,outputFile):
    
    tdata   = readCsv(testfile)
    n,m     = tdata.shape

    columns =  ['V%d' % number for number in range(1, m)]
    
    X_test  = tdata[columns]    
    X_test_n = copy.deepcopy(X_test)
    X_test_n  = normalize(X_test_n)

    Y_pred = model.predict(X_test_n)    

    Z = np.column_stack((X_test,Y_pred))
    Z = pd.DataFrame(Z, columns= ['V%d' % number for number in range(1, m)]+["Class"])
    Z.to_csv(outputFile)
    print("The prediction was written in: "+ outputFile)
    return Y_pred


def scorePrediction(scorefile,Y_pred):
    data  = readCsv(scorefile)
    target = 'Class'
    Y_test = data[target]
    print(classification_report(Y_test, Y_pred))


def main(argv):
    scorefile = ''
    if len(sys.argv) <= 6:
        print ('Use: predictLRMFD.py -i <trained_model> -t <csv_test_dataset> -o <csv_outputfile> -s <csv_score_dataset>')
        sys.exit(1)    
    try:
        opts, args = getopt.getopt(argv,"hi:t:o:s:",["ifile=","tfile=","ofile=","sfile="])
    except getopt.GetoptError:
        print ('Use: predictLRMFD.py -i <trained_model> -t <csv_test_dataset> -o <csv_outputfile> -s <csv_score_dataset>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('Use: predictLRMFD.py -i <trained_model> -t <csv_test_dataset> -o <csv_outputfile> -s <csv_score_dataset>') 
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-t", "--tfile"):
            testfile = arg            
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-s", "--sfile"):
            scorefile = arg     
            
    model = pickle.load(open(inputfile, 'rb'))    

    if model:
        Y_pred = makeAPrediction(model,testfile,outputfile)        
    else:
        print('Error: No model loaded.')
        sys.exit(1)

        
    if  len(scorefile.strip()) != 0:
        scorePrediction(scorefile,Y_pred)
        
                    
if __name__ == "__main__":
    main(sys.argv[1:])

    

    

