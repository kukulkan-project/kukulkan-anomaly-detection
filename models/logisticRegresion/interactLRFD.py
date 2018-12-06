#!/usr/bin/env python3

import logging
import sys, getopt
import pandas as pd
import numpy as np
import imp
#import readchar
#import cv2

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression()

inputfile = ''
outputfile = ''
testfile = ''

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
#        sys.exit(2)    
    except getopt.GetoptError as e:
        print ("Error:",filename, "error in reading." )
        print (e)     
        sys.exit(2)
    except pd.io.common.EmptyDataError:
        print ("Error: The file",filename,"is empty" )
        sys.exit(2)


def train(trainCsvFile):  
    data = readCsv(trainCsvFile);
    n,m = data.shape
       
    columns =  ['V%d' % number for number in range(1, m)]
    target = 'Class'

    X = data[columns]
    Y = data[target]
    X = normalize(X)

    model.fit(X, Y)
        

def makeAPrediction(testCsvFile,outputFile):
    #print(testCsvFile)
    #print(outputFile)
    tdata   = readCsv(testCsvFile)
    n,m     = tdata.shape
    columns =  ['V%d' % number for number in range(1, m)]
    
    X_test  = tdata[columns]    
    X_test  = normalize(X_test)

    Y_pred = model.predict(X_test)    
    Z = np.column_stack((X_test,Y_pred))
    Z = pd.DataFrame(Z, columns= ['V%d' % number for number in range(1, m)]+["Class"])
    Z.to_csv(outputFile)
    print("The prediction was written in: "+ outputFile)
    
        

def main(argv):

    if len(sys.argv) <= 4:
        print ('Use: interactLRFD.py -i <csv_train_dataset>  -o <outputfile>')
        sys.exit(1)    
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('Use: interactLRFD.py -i <csv_train_dataset>  -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('Use: interactLRFD.py -i <csv_train_dataset> -o <outputfile>')            
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            
    train(inputfile)
    
    exit = False
    while not exit:
        testfile = input('Give a csv test file, "q" to quit" \n')       
        if testfile=='q':
            exit=True            
        else:
            makeAPrediction(testfile,outputfile)        

                    
if __name__ == "__main__":
    main(sys.argv[1:])

    
