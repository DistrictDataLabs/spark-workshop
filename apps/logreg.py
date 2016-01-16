## Spark Application for performing logistic regression.
# Here, we're going to try some logistic regression to see how well we can separate red and white wines based on the measured features.
import csv

from numpy import array
from StringIO import StringIO

from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
# Load and parse the data

def parsePoint(tup):
    """
    Parse text data into floats.
    Return tuple of (label, features).
    """
    values = [float(x) for x in tup[1].split(';')]
    return LabeledPoint(tup[0], values[1:])

if __name__ == '__main__':
    conf = SparkConf().setAppName("Logistic Regression")
    sc   = SparkContext(conf=conf)

    ## load and parse the data
    # read in raw data
    print "Reading Data" + "\n"
    red_wines = sc.textFile("../fixtures/winequality-red.csv")
    white_wines = sc.textFile("../fixtures/winequality-white.csv")
    
    # separate the data from the headers
    red_wines_data = red_wines.zipWithIndex().filter(lambda s: s[1]>0).map(lambda s: s[0])
    white_wines_data = white_wines.zipWithIndex().filter(lambda s: s[1]>0).map(lambda s: s[0])
    
    # parse data and add labels
    labelled_white = white_wines_data.map(lambda s: (0, s)).map(parsePoint)
    labelled_red = red_wines_data.map(lambda s: (1,s)).map(parsePoint)
    
    # Combine into one RDD.
    all_wine = labelled_red.union(labelled_white)
    print "Training Model on " + str(all_wine.count()) + " data points " + "\n"
    
    # Build the model and train
    model = LogisticRegressionWithLBFGS.train(all_wine)

    # Evaluating the model on training data
    print "Evaluating Model " + "\n"
    labelsAndPreds = all_wine.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(all_wine.count())
    print("Training Error = " + str(trainErr)) + "\n"