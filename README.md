# Logistic Regression model for anomalies detection

## Software Requirements
    Python3
    
### Libraries   
    pandas, sklearn, sklearn2pmml and numpy
        
## Testing 
In models/logisticRegresion folder.
The next command create the train, test (1% size partition) and score cvs files for the model.

    ./splitLRMFD.py -i ../../databases/creditcard.csv -t train.csv -o test.csv -s score.csv -p .1


The next command build the trained Logistic Regression model with the data base 'train.csv'.   

       ./buildLRMFD.py -i train.csv -o lrmodel

## Client tests

The next command make a prediction  with the test.csv file and  score this prediction.

    ./predictLRMFD.py -i lrmodel.sav -t test.csv -o output.csv -s score.csv
    
The next command make a prediction  with the test.csv file.

    ./predictLRMFD.py -i lrmodel.sav -t test.csv -o output.csv

## Data Frame Structure

| V1 | V2 | ... |Vm| |Class|

Where Vi represents the features, and Class represents anomalies (0) or not (1), see the creditcard.csv file for reference.