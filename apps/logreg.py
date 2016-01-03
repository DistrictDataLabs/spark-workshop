## Spark Application for performing logistic regression.

import csv

from numpy import array
from StringIO import StringIO

from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel

# Load and parse the data

def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

if __name__ == '__main__':
    conf = SparkConf().setAppName("Logistic Regression")
    sc   = SparkContext(conf=conf)

    # load and parse the data
    data = sc.textFile("../fixtures/sample_svm_data.txt")
    parsedData = data.map(parsePoint)

    # Build the model and train
    model = LogisticRegressionWithLBFGS.train(parsedData)

    # Evaluating the model on training data
    labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
    print("Training Error = " + str(trainErr))