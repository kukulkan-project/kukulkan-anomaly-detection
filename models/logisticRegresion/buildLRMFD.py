#!/usr/bin/env python3

import logging
import sys, getopt
import pandas as pd
import numpy as np
import pickle
import imp

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline


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
        sys.exit(2)    
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
    #print(n)
    #print(m)
    
    columns =  ['V%d' % number for number in range(1, m-1)]
    target = 'Class'

    X = data[columns]
    Y = data[target]
    X = normalize(X)
    model.fit(X, Y)

    pipeline = PMMLPipeline([
	("mapper", DataFrameMapper([
	    ( ['V%d' % number for number in range(1, m-1)], [ContinuousDomain(), Imputer()])
	])),
	("classifier", model)
    ])
    pipeline.fit(X,Y)

    return pipeline
    
    

def main(argv):

    if len(sys.argv) <= 4:
        print ('Use: buildLRMFD.py -i <csv_train_dataset>  -o <name_model>')
        sys.exit(1)    
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('Use: buildLRMFD.py -t <csv_train_dataset>  -o <name_model>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('Use: buildLRMFD.py -i <csv_train_dataset> -o <name_model>') 
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            
    pipeline = train(inputfile)
    pickle.dump(model, open(outputfile+".sav", 'wb'))
    sklearn2pmml(pipeline, outputfile+".pmml", with_repr = True)
    
    print("\n Trained Logistic Regresion model for the data set " + inputfile+ " was saved in: " + outputfile+".sav and " + outputfile+".pmml")
            

                    
if __name__ == "__main__":
    main(sys.argv[1:])

    

    

